import argparse
import glob
import os
import statistics as st
import numpy as np
import torch

from sde_denoise.utils.ckpt import load_ckpt
from sde_denoise.utils.ema import EMA
from sde_denoise.scores.net_unet1d_dual import TinyUNet1D_Dual
from sde_denoise.sde.schedule import CosineSchedule
from sde_denoise.sde.additive_vp import AdditiveVP


def snr_db(clean, x):
    num = float((clean ** 2).sum()) + 1e-12
    den = float(((x - clean) ** 2).sum()) + 1e-12
    return 10.0 * np.log10(num / den)


def psnr_db(clean, x):
    peak = float(np.max(np.abs(clean))) + 1e-12
    mse  = float(np.mean((x - clean)**2)) + 1e-12
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--gamma_jump", type=float, default=0.5, help="strength of jump correction drift")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz in {args.folder}")

    device = args.device
    model = TinyUNet1D_Dual(in_ch=1, out_ch=1, t_dim=128).to(device)
    ema = EMA(model)
    load_ckpt(model, ema, opt=None, path=args.ckpt, map_location=device)
    ema.copy_to(model)
    model.eval()

    sch = CosineSchedule(device=device)
    sde = AdditiveVP(sch, device=device)

    def score_fn(x, t):
        sigma = sch.sigma(t).view(-1, 1, 1)
        eps_pred, _ = model(x, sch.log_snr(t))
        return -eps_pred / (sigma + 1e-12)

    def jump_corr_grad(x, t):
        x = x.detach().requires_grad_(True)
        lambda_t = sch.log_snr(t)
        _, logit = model(x, lambda_t)
        log_odds = logit.sum()
        g = torch.autograd.grad(log_odds, x, retain_graph=False, create_graph=False)[0]
        x = x.detach()
        return g

    mse_n, mse_d, sin_list, sout_list, pin_list, pout_list = [], [], [], [], [], []

    for path in files:
        data = np.load(path, allow_pickle=True)
        if "clean" not in data or "noisy" not in data:
            continue

        clean = data["clean"].astype(np.float32)
        noisy = data["noisy"].astype(np.float32)

        y = torch.from_numpy(noisy[None, None, :]).to(device)

        x = torch.randn_like(y)
        B = 1
        t = torch.full((B,), 1.0 - 1e-8, device=device)
        dt = 1.0 / args.steps

        for _ in range(args.steps):
            with torch.enable_grad():
                prior_v = sde.pf_ode_vecfield(x, t, score_fn)
                jg = args.gamma_jump * jump_corr_grad(x, t)
                v1 = prior_v + jg
                k1 = v1
                x_mid = x + dt * k1
                t_mid = t - dt

                prior_v2 = sde.pf_ode_vecfield(x_mid, t_mid, score_fn)
                jg2 = args.gamma_jump * jump_corr_grad(x_mid, t_mid)
                v2 = prior_v2 + jg2

            x = x + 0.5 * dt * (k1 + v2)
            t = t_mid
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-1e6, 1e6)

        xhat = x.detach().squeeze().cpu().numpy().astype(np.float32)

        mse_n.append(float(np.mean((noisy - clean) ** 2)))
        mse_d.append(float(np.mean((xhat - clean) ** 2)))
        sin_list.append(snr_db(clean, noisy))
        sout_list.append(snr_db(clean, xhat))
        pin_list.append(psnr_db(clean, noisy))
        pout_list.append(psnr_db(clean, xhat))

    print("Files evaluated:", len(mse_d))
    if mse_d:
        print(f"MSE (noisy):   mean={st.mean(mse_n):.4e}")
        print(f"MSE (denoised):mean={st.mean(mse_d):.4e}")
        print(f"SNR_in (dB):   mean={st.mean(sin_list):.2f}")
        print(f"SNR_out (dB):  mean={st.mean(sout_list):.2f}")
        print(f"ΔSNR (dB):     mean={st.mean([o - i for i, o in zip(sin_list, sout_list)]):.2f}")
        print(f"PSNR_in (dB):  mean={st.mean(pin_list):.2f}")
        print(f"PSNR_out (dB): mean={st.mean(pout_list):.2f}")
        print(f"ΔPSNR (dB):    mean={st.mean([o - i for i, o in zip(pin_list, pout_list)]):.2f}")


if __name__ == "__main__":
    main()