import argparse, torch
from omegaconf import OmegaConf
from sde_denoise.train.dataset import NPZFolder1D
from sde_denoise.scores.net_unet1d import TinyUNet1D
from sde_denoise.train.engine import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="synthetic_data")
    ap.add_argument("--noise_kind", default=None, help="override config")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.noise_kind is not None:
        cfg.noise_kind = args.noise_kind

    ds = NPZFolder1D(root=args.root, noise_kind=cfg.noise_kind)
    model = TinyUNet1D(in_ch=1, out_ch=1, t_dim=128)
    tr = Trainer(cfg, ds, model, device=args.device)
    tr.train(epochs=args.epochs)

if __name__ == "__main__":
    main()
