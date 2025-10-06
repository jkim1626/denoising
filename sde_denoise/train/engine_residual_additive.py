import os
import torch
import tqdm
from torch.utils.data import DataLoader

from ..utils.ema import EMA
from ..utils.ckpt import save_ckpt
from ..utils.logging import get_logger
from ..sde.schedule import CosineSchedule
from ..sde.additive_vp import AdditiveVP


def _per_sample_residual_loss_additive(model, sch, x0, t):
    """
    Semi-oracle residualization for additive VP.

    Forward: x_t = a x0 + s ε
    Analytic conditional score: s_analytic = -(x_t - a x0) / s^2
    Model head predicts ε; convert to score: s_pred = -ε_pred / s
    Train the model to match s_analytic (i.e., learn residual around it):
        L = || s_pred - s_analytic ||^2  (per-sample, reduced over C,L)
    """
    a = sch.alpha_bar(t).view(-1, 1, 1)
    s = sch.sigma(t).view(-1, 1, 1)
    eps = torch.randn_like(x0)
    xt = a * x0 + s * eps

    s_analytic = -(xt - a * x0) / (s ** 2 + 1e-12)

    eps_pred = model(xt, sch.log_snr(t))
    s_pred = -eps_pred / (s + 1e-12)

    per = torch.mean((s_pred - s_analytic) ** 2, dim=(1, 2))  # (B,)
    return per


class TrainerResidualAdditive:
    """
    Trainer that learns residual score on top of the analytic Gaussian conditional score.
    """
    def __init__(self, cfg, dataset, model, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.logger = get_logger()

        self.sch = CosineSchedule(device=device)
        self.sde = AdditiveVP(self.sch, device=device)

        self.model = model.to(device)
        self.ema = EMA(self.model, decay=float(cfg.get("ema", 0.999)))
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.get("lr", 2e-4)),
            weight_decay=1e-4,
        )

        self.dl = DataLoader(
            dataset,
            batch_size=int(cfg.get("batch", 16)),
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        self.run_dir = cfg.get("run_dir", "runs/additive_residual")
        os.makedirs(self.run_dir, exist_ok=True)

    def _sample_t(self, B):
        return torch.rand(B, device=self.device).clamp(1e-4, 1 - 1e-4)

    def train(self, epochs=10):
        self.model.train()
        for ep in range(epochs):
            pbar = tqdm.tqdm(self.dl, desc=f"[RES] Epoch {ep+1}/{epochs}")
            for (x0, _) in pbar:
                x0 = x0.to(self.device)
                B = x0.size(0)
                t = self._sample_t(B)

                per = _per_sample_residual_loss_additive(self.model, self.sch, x0, t)
                loss = per.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.ema.update(self.model)

                pbar.set_postfix(loss=float(loss.item()))

            ck = os.path.join(self.run_dir, "latest.pt")
            save_ckpt(self.model, self.ema, self.opt, ck)
