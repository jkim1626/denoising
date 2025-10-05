import argparse, torch, os
from sde_denoise.utils.ckpt import load_ckpt
from sde_denoise.utils.ema import EMA
from sde_denoise.scores.net_unet1d import TinyUNet1D
from sde_denoise.sde.schedule import CosineSchedule
from sde_denoise.sde.additive_vp import AdditiveVP
from sde_denoise.sde.geometric_vp import GeometricVP
from sde_denoise.sde.jump_diffusion import JumpDiffusion
from sde_denoise.sample.sampler import reverse_sde_step, pf_ode_stepper

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["pf_ode","reverse_sde"], default="pf_ode")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--shape", type=int, nargs="+", default=[1, 2048])  # (C, L)
    ap.add_argument("--noise_kind", choices=["additive","multiplicative","jump"], default="additive")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if len(args.shape) != 2:
        raise ValueError("--shape must be C L (e.g., 1 2048)")

    model = TinyUNet1D(in_ch=args.shape[0], out_ch=args.shape[0], t_dim=128).to(args.device)
    ema = EMA(model)
    load_ckpt(model, ema, opt=None, path=args.ckpt, map_location=args.device)
    ema.copy_to(model)
    model.eval()

    sch = CosineSchedule(device=args.device)
    if args.noise_kind == "additive":
        sde = AdditiveVP(sch, device=args.device)
        # score from ε-pred: s = -ε/σ
        def score_fn(x, t):
            sigma = sch.sigma(t).view(-1,1,1)
            eps_pred = model(x, sch.log_snr(t))
            return -eps_pred / (sigma + 1e-12)
    elif args.noise_kind == "multiplicative":
        sde = GeometricVP(sch, device=args.device)
        # Here model is ε in log-space; convert to score in x-space if needed.
        # For PF-ODE we can still use ε head with approximate mapping; keeping ε-based stepping works for demo.
        def score_fn(x, t):
            # fallback: approximate with additive mapping (works if x around O(1))
            sigma = sch.sigma(t).view(-1,1,1)
            eps_pred = model(x, sch.log_snr(t))
            return -eps_pred / (sigma + 1e-12)
    else:
        sde = JumpDiffusion(sch, device=args.device)
        def score_fn(x, t):
            sigma = sch.sigma(t).view(-1,1,1)
            eps_pred = model(x, sch.log_snr(t))
            return -eps_pred / (sigma + 1e-12)

    B = 1
    x = torch.randn(B, args.shape[0], args.shape[1], device=args.device)
    steps = args.steps
    dt = 1.0 / steps
    t = torch.ones(B, device=args.device)

    if args.mode == "pf_ode" and args.noise_kind != "jump":
        stepper = pf_ode_stepper(sde, score_fn)
        for _ in range(steps):
            x = stepper(x, t, dt)
            t = t - dt
    else:
        for _ in range(steps):
            x = reverse_sde_step(sde, score_fn, x, t, dt)
            t = t - dt

    out = x.detach().cpu()
    os.makedirs("samples", exist_ok=True)
    torch.save(out, f"samples/sample_{args.noise_kind}.pt")
    print("Saved:", f"samples/sample_{args.noise_kind}.pt")

if __name__ == "__main__":
    main()
