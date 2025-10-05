import argparse, os, json, re
import numpy as np
import torch
from sde_denoise.utils.ckpt import load_ckpt
from sde_denoise.utils.ema import EMA
from sde_denoise.scores.net_unet1d import TinyUNet1D
from sde_denoise.sde.schedule import CosineSchedule
from sde_denoise.sde.additive_vp import AdditiveVP

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--snr_db", type=float, default=None, help="override SNR in dB if not in meta/filename")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # --- load data ---
    d = np.load(args.npz, allow_pickle=True)
    if "noisy" in d:
        x_noisy = d["noisy"].astype(np.float32)
    elif "clean" in d:  # fallback if no noisy data
        x_noisy = (d["clean"].astype(np.float32) + 0.01*np.random.randn(*d["clean"].shape)).astype(np.float32)
    else:
        raise KeyError("npz must contain 'noisy' or 'clean'.")

    # --- infer SNR from metadata or filename ---
    snr_db = args.snr_db
    if snr_db is None and "meta_json" in d:
        try:
            meta = json.loads(str(d["meta_json"]))
            if "snr_db" in meta:
                snr_db = float(meta["snr_db"])
            elif "snr" in meta:
                snr_db = float(meta["snr"])
        except Exception:
            pass
    if snr_db is None:
        m = re.search(r"snr(-?\d+(\.\d+)?)", os.path.basename(args.npz))
        if m:
            snr_db = float(m.group(1))
    if snr_db is None:
        print("WARNING: SNR(dB) not found; defaulting to 10 dB.")
        snr_db = 10.0

    x = torch.from_numpy(x_noisy[None, None, :]).to(args.device)  # (1,1,L)

    # --- load model ---
    model = TinyUNet1D(in_ch=1, out_ch=1, t_dim=128).to(args.device)
    ema = EMA(model)
    load_ckpt(model, ema, opt=None, path=args.ckpt, map_location=args.device)
    ema.copy_to(model)
    model.eval()

    sch = CosineSchedule(device=args.device)
    sde = AdditiveVP(sch, device=args.device)

    # --- estimate t* based on SNR ---
    r2 = 10.0 ** (-snr_db / 10.0)
    grid = torch.linspace(1e-5, 1.0 - 1e-5, 2000, device=args.device)
    a = sch.alpha_bar(grid)
    s = sch.sigma(grid)
    ratio2 = (s / a) ** 2
    idx = torch.argmin(torch.abs(ratio2 - r2))
    t_star = grid[idx].detach()

    def score_fn(x, t):
        sigma = sch.sigma(t).view(-1, 1, 1)
        eps_pred = model(x, sch.log_snr(t))
        return -eps_pred / (sigma + 1e-12)

    s_hat = score_fn(x, t_star.expand(x.size(0)))
    a_star = sch.alpha_bar(t_star).view(1, 1, 1)
    s_star = sch.sigma(t_star).view(1, 1, 1)
    x0_hat = (x + (s_star ** 2) * s_hat) / (a_star + 1e-12)

    # --- save to "denoised" subfolder ---
    # Example: synthetic_data/additive/denoised/
    base_dir = os.path.dirname(args.npz)
    denoised_dir = os.path.join(base_dir, "denoised")
    os.makedirs(denoised_dir, exist_ok=True)

    base_name = os.path.basename(args.npz)
    out_path = os.path.join(denoised_dir, base_name.replace(".npz", "_denoised.npz"))
    np.savez(out_path, denoised=x0_hat.squeeze().cpu().numpy().astype(np.float32))

    print(f"Saved denoised file to:\n  {out_path}")

if __name__ == "__main__":
    main()
