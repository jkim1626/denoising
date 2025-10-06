import argparse, os, json, re
import numpy as np
import torch

from sde_denoise.utils.ckpt import load_ckpt
from sde_denoise.utils.ema import EMA
from sde_denoise.scores.net_unet1d import TinyUNet1D
from sde_denoise.sde.schedule import CosineSchedule

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_scale", type=float, default=1.0)
    ap.add_argument("--snr_db", type=float, default=None, help="optional; used to pick t*")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    if "noisy" not in d:
        raise KeyError("npz must contain 'noisy'.")
    x_noisy = d["noisy"].astype(np.float32)
    if (x_noisy <= 0).any():
        # shift into positive domain minimally
        shift = float(max(1e-6, -x_noisy.min() + 1e-6))
        x_noisy = x_noisy + shift

    # model
    model = TinyUNet1D(in_ch=1, out_ch=1, t_dim=128).to(args.device)
    ema = EMA(model); load_ckpt(model, ema, opt=None, path=args.ckpt, map_location=args.device)
    ema.copy_to(model); model.eval()
    sch = CosineSchedule(device=args.device)

    # choose t* by matching (σ/α)^2 ≈ 1/SNR in LOG space
    snr_db = args.snr_db
    if snr_db is None and "meta_json" in d:
        try:
            meta = json.loads(str(d["meta_json"]))
            snr_db = float(meta.get("snr_db", meta.get("snr", 10.0)))
        except Exception:
            snr_db = None
    if snr_db is None:
        m = re.search(r"snr(-?\d+(\.\d+)?)", os.path.basename(args.npz))
        snr_db = float(m.group(1)) if m else 10.0

    r2 = 10.0**(-snr_db/10.0)
    grid = torch.linspace(1e-5, 1.0-1e-5, 2000, device=args.device)
    a = sch.alpha_bar(grid); s = sch.sigma(grid)
    t_star = grid[torch.argmin(torch.abs((s/a)**2 - r2))]

    x = torch.from_numpy(x_noisy[None, None, :]).to(args.device)
    y = torch.log(torch.clamp(x, min=1e-6))

    B = x.size(0)
    lambda_t = sch.log_snr(t_star).repeat(B)

    eps_pred = model(y, lambda_t)
    sigma = sch.sigma(t_star).view(1,1,1)
    alpha = sch.alpha_bar(t_star).view(1,1,1)
    # score in LOG space
    score_y = -args.score_scale * eps_pred / (sigma + 1e-12)
    # Tweedie posterior mean in log space
    y0_hat = (y + (sigma**2) * score_y) / (alpha + 1e-12)
    x0_hat = torch.exp(y0_hat).squeeze().cpu().numpy().astype(np.float32)

    out_dir = os.path.join(os.path.dirname(args.npz), "denoised_mult_logvp")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(args.npz).replace(".npz", "_mult_logvp_denoised.npz")
    out_path = os.path.join(out_dir, base)
    np.savez(out_path, denoised=x0_hat)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
