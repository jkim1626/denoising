import argparse, glob, os, json, re
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
    ap.add_argument("--folder", default="synthetic_data/additive")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No npz in {args.folder}")

    model = TinyUNet1D(in_ch=1, out_ch=1, t_dim=128).to(args.device)
    ema = EMA(model)
    load_ckpt(model, ema, opt=None, path=args.ckpt, map_location=args.device)
    ema.copy_to(model)
    model.eval()

    sch = CosineSchedule(device=args.device)
    sde = AdditiveVP(sch, device=args.device)

    def score_fn(x, t):
        sigma = sch.sigma(t).view(-1,1,1)
        eps_pred = model(x, sch.log_snr(t))
        return -eps_pred / (sigma + 1e-12)

    def file_snr_db(clean, noisy):
        num = np.sum(clean**2) + 1e-12
        den = np.sum((noisy-clean)**2) + 1e-12
        return 10.0 * np.log10(num/den)

    def psnr_db(clean, x):
        peak = float(np.max(np.abs(clean))) + 1e-12
        mse  = float(np.mean((x - clean)**2)) + 1e-12
        return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)

    m_mse_noisy, m_mse_deno, m_snr_in, m_snr_out, m_psnr_in, m_psnr_out = [], [], [], [], [], []

    for p in files:
        d = np.load(p, allow_pickle=True)
        if "clean" not in d or "noisy" not in d:
            continue
        x0 = d["clean"].astype(np.float32)
        xn = d["noisy"].astype(np.float32)
        snr_in = file_snr_db(x0, xn)
        r2 = 10.0 ** (-snr_in / 10.0)
        grid = torch.linspace(1e-5, 1.0-1e-5, 2000, device=args.device)
        a = sch.alpha_bar(grid); s = sch.sigma(grid)
        ratio2 = (s / a)**2
        idx = torch.argmin(torch.abs(ratio2 - r2))
        t_star = grid[idx].detach()

        xt = torch.from_numpy(xn[None,None,:]).to(args.device)
        s_hat = score_fn(xt, t_star.expand(1))
        a_star = sch.alpha_bar(t_star).view(1,1,1)
        s_star = sch.sigma(t_star).view(1,1,1)
        xhat = (xt + (s_star**2) * s_hat) / (a_star + 1e-12)
        xhat = xhat.squeeze().cpu().numpy()

        mse_noisy = float(np.mean((xn - x0)**2))
        mse_deno  = float(np.mean((xhat - x0)**2))
        snr_out   = file_snr_db(x0, xhat)

        m_mse_noisy.append(mse_noisy)
        m_mse_deno.append(mse_deno)
        m_snr_in.append(snr_in)
        m_snr_out.append(snr_out)
        m_psnr_in.append(psnr_db(x0, xn))
        m_psnr_out.append(psnr_db(x0, xhat))

    if not m_mse_deno:
        print("No evaluable files found (need both clean & noisy).")
        return

    import statistics as stats
    print(f"Files evaluated: {len(m_mse_deno)}")
    print(f"MSE (noisy):   mean={stats.mean(m_mse_noisy):.4e}")
    print(f"MSE (denoised):mean={stats.mean(m_mse_deno):.4e}")
    print(f"SNR_in (dB):   mean={stats.mean(m_snr_in):.2f}")
    print(f"SNR_out (dB):  mean={stats.mean(m_snr_out):.2f}")
    print(f"ΔSNR (dB):     mean={stats.mean([o-i for i,o in zip(m_snr_in,m_snr_out)]):.2f}")
    print(f"PSNR_in (dB):  mean={stats.mean(m_psnr_in):.2f}")
    print(f"PSNR_out (dB): mean={stats.mean(m_psnr_out):.2f}")
    print(f"ΔPSNR (dB):    mean={stats.mean([o-i for i,o in zip(m_psnr_in,m_psnr_out)]):.2f}")

if __name__ == "__main__":
    main()