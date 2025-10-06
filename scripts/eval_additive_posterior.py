import argparse, glob, os, json, re, statistics as st
import numpy as np
import torch
from omegaconf import OmegaConf

from sde_denoise.utils.ckpt import load_ckpt
from sde_denoise.utils.ema import EMA
from sde_denoise.scores.net_unet1d import TinyUNet1D
from sde_denoise.sde.schedule import CosineSchedule
from sde_denoise.sde.additive_vp import AdditiveVP

def robust_var(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return float((1.4826 * mad) ** 2 + 1e-12)

def safe_snr_db(clean, x):
    num = float((clean**2).sum()) + 1e-12
    den = float(((x-clean)**2).sum()) + 1e-12
    return 10.0 * np.log10(num / den)

def psnr_db(clean, x):
    peak = float(np.max(np.abs(clean))) + 1e-12
    mse  = float(np.mean((x - clean)**2)) + 1e-12
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)

def rms_torch(x):
    return torch.sqrt(torch.mean(x**2) + 1e-12)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ckpt        = cfg.get("ckpt", "runs/additive/latest.pt")
    folder      = cfg.get("folder", "synthetic_data/additive")
    steps       = int(cfg.get("steps", 100))
    like_weight = float(cfg.get("like_weight", 0.3))
    score_scale = float(cfg.get("score_scale", 1.0))

    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    device = args.device
    model = TinyUNet1D(in_ch=1, out_ch=1, t_dim=128).to(device)
    ema = EMA(model)
    load_ckpt(model, ema, opt=None, path=ckpt, map_location=device)
    ema.copy_to(model)
    model.eval()

    sch = CosineSchedule(device=device)
    sde = AdditiveVP(sch, device=device)

    def score_fn(x, t):
        sigma = sch.sigma(t).view(-1,1,1)
        eps_pred = model(x, sch.log_snr(t))
        return -score_scale * eps_pred / (sigma + 1e-12)

    out_dir = os.path.join(folder, "posterior_denoised")
    os.makedirs(out_dir, exist_ok=True)

    mse_noisy, mse_deno, snr_in_list, snr_out_list, psnr_in_list, psnr_out_list = [], [], [], [], [], []
    n_proc = 0

    for p in files:
        d = np.load(p, allow_pickle=True)
        if "clean" not in d or "noisy" not in d:
            continue

        clean = d["clean"].astype(np.float32)
        noisy = d["noisy"].astype(np.float32)

        snr_db_meta = None
        if "meta_json" in d:
            try:
                meta = json.loads(str(d["meta_json"]))
                if "snr_db" in meta: snr_db_meta = float(meta["snr_db"])
                elif "snr" in meta:  snr_db_meta = float(meta["snr"])
            except Exception:
                pass
        if snr_db_meta is None:
            m = re.search(r"snr(-?\d+(\.\d+)?)", os.path.basename(p))
            if m: snr_db_meta = float(m.group(1))
        if snr_db_meta is None:
            snr_db_meta = 10.0

        y = noisy
        y_var = robust_var(y)
        snr_lin = 10.0 ** (snr_db_meta / 10.0)
        tau2_est = y_var / snr_lin
        tau2_floor = max(1e-6 * y_var, 1e-8)
        tau2 = max(tau2_est, tau2_floor)

        y_t = torch.from_numpy(y[None, None, :]).to(device)
        x = torch.randn_like(y_t)
        B = 1
        t = torch.full((B,), 1.0 - 1e-8, device=device)
        dt = 1.0 / steps

        g_rms_cap = 3.0

        for _ in range(steps):
            prior_v = sde.pf_ode_vecfield(x, t, score_fn)

            sigma_t2 = sch.sigma(t).view(-1,1,1)**2
            like_raw = (y_t - x) / (tau2 + 1e-12)
            g_r = rms_torch(like_raw)
            scale = min(1.0, g_rms_cap / float(g_r.item())) if torch.isfinite(g_r) else 1.0
            like_g = scale * sigma_t2 * like_raw

            v1 = prior_v + like_weight * like_g
            k1 = v1
            x_mid = x + dt * k1
            t_mid = t - dt

            prior_v2 = sde.pf_ode_vecfield(x_mid, t_mid, score_fn)
            sigma_mid2 = sch.sigma(t_mid).view(-1,1,1)**2
            like_raw2 = (y_t - x_mid) / (tau2 + 1e-12)
            g_r2 = rms_torch(like_raw2)
            scale2 = min(1.0, g_rms_cap / float(g_r2.item())) if torch.isfinite(g_r2) else 1.0
            like_g2 = scale2 * sigma_mid2 * like_raw2

            v2 = prior_v2 + like_weight * like_g2
            x = x + 0.5 * dt * (k1 + v2)
            t = t_mid

            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-1e6, 1e6)

        xhat = x.squeeze().cpu().numpy().astype(np.float32)

        base = os.path.basename(p).replace(".npz", "_posterior_denoised.npz")
        out_path = os.path.join(out_dir, base)
        np.savez(out_path, denoised=xhat, meta={"snr_db": float(snr_db_meta),
                                                "tau2": float(tau2),
                                                "score_scale": float(score_scale),
                                                "like_weight": float(like_weight)})

        mse_noisy.append(float(np.mean((noisy - clean)**2)))
        mse_deno.append(float(np.mean((xhat - clean)**2)))
        snr_in_list.append(safe_snr_db(clean, noisy))
        snr_out_list.append(safe_snr_db(clean, xhat))
        psnr_in_list.append(psnr_db(clean, noisy))
        psnr_out_list.append(psnr_db(clean, xhat))
        n_proc += 1

    print("Files evaluated:", n_proc)
    if n_proc == 0:
        print("No evaluable files (need both 'clean' and 'noisy').")
        return
    print(f"MSE (noisy):   mean={st.mean(mse_noisy):.4e}")
    print(f"MSE (denoised):mean={st.mean(mse_deno):.4e}")
    print(f"SNR_in (dB):   mean={st.mean(snr_in_list):.2f}")
    print(f"SNR_out (dB):  mean={st.mean(snr_out_list):.2f}")
    print(f"ΔSNR (dB):     mean={st.mean([o-i for i,o in zip(snr_in_list,snr_out_list)]):.2f}")
    print(f"PSNR_in (dB):  mean={st.mean(psnr_in_list):.2f}")
    print(f"PSNR_out (dB): mean={st.mean(psnr_out_list):.2f}")
    print(f"ΔPSNR (dB):    mean={st.mean([o-i for i,o in zip(psnr_in_list,psnr_out_list)]):.2f}")
    print(f"Outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()