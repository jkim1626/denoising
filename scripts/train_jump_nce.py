import argparse
import torch
from omegaconf import OmegaConf

from sde_denoise.train.dataset import NPZFolder1D
from sde_denoise.scores.net_unet1d_dual import TinyUNet1D_Dual
from sde_denoise.train.engine_jump_nce import TrainerJumpNCE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="synthetic_data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ds = NPZFolder1D(root=args.root, noise_kind="jump")  # expects synthetic_data/jump/*.npz
    model = TinyUNet1D_Dual(in_ch=1, out_ch=1, t_dim=128)

    trainer = TrainerJumpNCE(cfg, ds, model, device=args.device)
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
