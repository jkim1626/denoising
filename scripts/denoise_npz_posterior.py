import argparse, os, json, re, numpy as np, torch
from sde_denoise.utils.ckpt import load_ckpt
from sde_denoise.utils.ema import EMA
from sde_denoise.scores.net_unet1d import TinyUNet1D
from sde_denoise.sde.schedule import CosineSchedule
from sde_denoise.sde.additive_vp import AdditiveVP
from sde_denoise.sample.sampler_posterior import pf_ode_stepper_posterior

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--score_scale", type=float, default=1.0, help="global score calibration")
    ap.add_argument("--like_weight", type=float, default=1.0, help="likelihood gradient weight")
    ap.add_argument("--snr_db", type=float, default=None, help="SNR of y; infers if not provided")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    if "noisy" in d:
        y = d["noisy"].astype(np.float32)
    elif "clean" in d:
        y = (d["clean"].astype(np.float32) + 0.01*np.random.randn(*d["clean"].shape)).astype(np.float32)
    else:
        raise KeyError("npz must contain 'noisy' or 'clean'.")

    # infer SNR(dB)
    snr_db = args.snr_db
    if snr_db is None and "meta_json" in d:
        try:
            meta = json.loads(str(d["meta_json"]))
            if "snr_db" in meta: snr_db = float(meta["snr_db"])
            elif "snr" in meta: snr_db = float(meta["snr"])
        except Exception: pass
    if snr_db is None:
        m = re.search(r"snr(-?\d+(\.\d+)?)", os.path.basename(args.npz))
        if m: snr_db = float(m.group(1))
    if snr_db is None:
        snr_db = 10.0  # fallback

    y_t = torch.from_numpy(y[None, None, :]).to(args.device)

    # model
    model = TinyUNet1D(in_ch=1, out_ch=1, t_dim=128).to(args.device)
    ema = EMA(model)
    load_ckpt(model, ema, opt=None, path=args.ckpt, map_location=args.device)
    ema.copy_to(model)
    model.eval()

    sch = CosineSchedule(device=args.device)
    sde = AdditiveVP(sch, device=args.device)

    # prior score from epsilon head
    def score_fn(x, t):
        sigma = sch.sigma(t).view(-1,1,1)
        eps_pred = model(x, sch.log_snr(t))
        return -eps_pred / (sigma + 1e-12)

    # Gaussian likelihood with identity operator: p(y|x) ~ N(y; x, tau^2 I)
    # Estimate tau^2 from SNR: SNR_linear = ||x0||^2 / ||n||^2 ≈ 10^(snr_db/10)
    # We'll approximate tau^2 by var(y)/SNR_lin. This is crude but effective on synthetic data.
    y_var = float(np.var(y) + 1e-12)
    snr_lin = 10.0 ** (snr_db / 10.0)
    tau2 = y_var / snr_lin

    def like_grad_fn(x, t):
        # ∇_x log N(y; x, tau^2 I) = (y - x) / tau^2
        return (y_t - x) / (tau2 + 1e-12)

    # integrate PF-ODE with posterior tilt
    B = 1
    x = torch.randn_like(y_t)  # initialization from prior at t=1
    steps = args.steps
    dt = 1.0 / steps
    t = torch.full((B,), 1.0 - 1e-8, device=args.device)

    stepper = pf_ode_stepper_posterior(sde, score_fn, like_grad_fn,
                                       like_weight=args.like_weight,
                                       score_scale=args.score_scale)
    for _ in range(steps):
        x = stepper(x, t, dt)
        t = t - dt

    xhat = x.squeeze().cpu().numpy().astype(np.float32)

    # save next to input in a 'posterior_denoised' subfolder
    base_dir = os.path.dirname(args.npz)
    out_dir = os.path.join(base_dir, "posterior_denoised")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(args.npz).replace(".npz", "_posterior_denoised.npz")
    out_path = os.path.join(out_dir, base)
    np.savez(out_path, denoised=xhat, meta={"snr_db": snr_db, "tau2": tau2,
                                            "score_scale": float(args.score_scale),
                                            "like_weight": float(args.like_weight)})
    print("Saved posterior-denoised:", out_path)

if __name__ == "__main__":
    main()
