import argparse, os, json, re
import numpy as np
import torch

from sde_denoise.utils.ckpt import load_ckpt
from sde_denoise.utils.ema import EMA
from sde_denoise.scores.net_unet1d import TinyUNet1D
from sde_denoise.sde.schedule import CosineSchedule
from sde_denoise.sde.geometric_vp import GeometricVP

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--snr_db", type=float, default=None, help="optional heuristic for choosing t*")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    if "noisy" in d:
        x_noisy = d["noisy"].astype(np.float32)
    else:
        raise KeyError("npz must contain 'noisy' for multiplicative denoising.")

    # --- model ---
    model = TinyUNet1D(in_ch=1, out_ch=1, t_dim=128).to(args.device)
    ema = EMA(model)
    load_ckpt(model, ema, opt=None, path=args.ckpt, map_location=args.device)
    ema.copy_to(model); model.eval()

    sch = CosineSchedule(device=args.device)
    _ = GeometricVP(sch, device=args.device)  # reserved for later extensions

    # --- choose t* (heuristic) ---
    snr_db = args.snr_db
    if snr_db is None and "meta_json" in d:
        try:
            meta = json.loads(str(d["meta_json"]))
            snr_db = float(meta.get("snr_db", meta.get("snr", 10.0)))
        except Exception:
            snr_db = None
    if snr_db is None:
        m = re.search(r"snr(-?\d+(\.\d+)?)", os.path.basename(args.npz))
        snr_db = float(m.group(1)) if m else None

    grid = torch.linspace(1e-5, 1 - 1e-5, 2000, device=args.device)
    Lam = sch.Lambda(grid)

    if snr_db is not None:
        target_val = 1.0 / (1.0 + 10.0 ** (snr_db / 10.0))  # float
        lam_max = float(Lam.max().detach().cpu().item())
        target_val = max(1e-4, min(lam_max, target_val))
        target_Lam = torch.tensor(target_val, device=args.device)
    else:
        target_Lam = Lam[len(grid) // 2]

    tidx = torch.argmin(torch.abs(Lam - target_Lam))
    t_star = grid[tidx]

    # --- Tweedie in log-domain ---
    x = torch.from_numpy(x_noisy[None, None, :]).to(args.device)  # (B=1,1,L)
    y = torch.log(torch.clamp(x.abs(), min=1e-6))

    Lam_star = sch.Lambda(t_star).view(1, 1, 1)
    std_star = torch.sqrt(torch.clamp(Lam_star, min=1e-12))

    # FIX: expand lambda_t to (B,) for time embedding
    B = x.size(0)
    lambda_t = sch.log_snr(t_star).repeat(B)  # shape (B,)

    eps_pred = model(y, lambda_t)                  # predict eps in log-domain
    score_y = -eps_pred / (std_star + 1e-12)      # convert to score
    y0_hat = y + Lam_star * score_y               # Tweedie posterior mean in log-domain
    x0_hat = torch.exp(y0_hat).squeeze().cpu().numpy().astype(np.float32)

    out_dir = os.path.join(os.path.dirname(args.npz), "denoised_mult")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(args.npz).replace(".npz", "_mult_denoised.npz")
    out_path = os.path.join(out_dir, base)
    np.savez(out_path, denoised=x0_hat)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
