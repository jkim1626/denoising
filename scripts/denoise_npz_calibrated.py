import argparse, os, json, re, numpy as np, torch
from sde_denoise.utils.ckpt import load_ckpt
from sde_denoise.utils.ema import EMA
from sde_denoise.scores.net_unet1d import TinyUNet1D
from sde_denoise.sde.schedule import CosineSchedule

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--score_scale", type=float, default=1.0)
    ap.add_argument("--snr_db", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    x_noisy = d["noisy"].astype(np.float32) if "noisy" in d else d["clean"].astype(np.float32)

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

    model = TinyUNet1D(in_ch=1, out_ch=1, t_dim=128).to(args.device)
    ema = EMA(model); load_ckpt(model, ema, opt=None, path=args.ckpt, map_location=args.device); ema.copy_to(model); model.eval()
    sch = CosineSchedule(device=args.device)

    r2 = 10.0**(-snr_db/10.0)
    grid = torch.linspace(1e-5, 1.0-1e-5, 2000, device=args.device)
    a = sch.alpha_bar(grid); s = sch.sigma(grid)
    t_star = grid[torch.argmin(torch.abs((s/a)**2 - r2))]

    x = torch.from_numpy(x_noisy[None, None, :]).to(args.device)
    sigma = sch.sigma(t_star).view(1,1,1)
    alpha = sch.alpha_bar(t_star).view(1,1,1)

    lambda_t = sch.log_snr(t_star).repeat(x.size(0))
    eps_pred = model(x, lambda_t)
    score = -args.score_scale * eps_pred / (sigma + 1e-12)
    x0_hat = (x + (sigma**2) * score) / (alpha + 1e-12)

    out_dir = os.path.join(os.path.dirname(args.npz), f"denoised_cal")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(args.npz).replace(".npz", f"_cal{args.score_scale:.2f}.npz"))
    np.savez(out, denoised=x0_hat.squeeze().cpu().numpy().astype(np.float32))
    print("Saved:", out)

if __name__ == "__main__":
    main()
