import os
import torch
import tqdm
from torch.utils.data import DataLoader

from ..sde.schedule import CosineSchedule
from ..sde.additive_vp import AdditiveVP
from ..utils.ema import EMA
from ..utils.ckpt import save_ckpt
from ..utils.logging import get_logger


def sample_forward_xt(sch, x0, t):
    a = sch.alpha_bar(t).view(-1, 1, 1)
    s = sch.sigma(t).view(-1, 1, 1)
    eps = torch.randn_like(x0)
    return a * x0 + s * eps


def jump_corrupt(x, p=0.01, amp=3.0):
    """
    Simple impulsive corruption: with probability p at each position,
    add a large-amplitude spike ~ N(0, amp^2).
    """
    mask = (torch.rand_like(x) < p).float()
    spikes = amp * torch.randn_like(x)
    return x + mask * spikes


class TrainerJumpNCE:
    """
    Learn a density-ratio via NCE between:
      - positives: x_t sampled from forward (clean) kernel
      - negatives: jump-corrupted x_t
    Optional auxiliary eps-DSM loss keeps representation sane.
    """
    def __init__(self, cfg, dataset, model_dual, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.logger = get_logger()

        self.sch = CosineSchedule(device=device)
        self.sde = AdditiveVP(self.sch, device=device)

        self.model = model_dual.to(device)
        self.ema = EMA(self.model, decay=float(cfg.get("ema", 0.999)))
        self.opt = torch.optim.AdamW(self.model.parameters(),
                                     lr=float(cfg.get("lr", 2e-4)),
                                     weight_decay=1e-4)

        self.dl = DataLoader(dataset,
                             batch_size=int(cfg.get("batch", 16)),
                             shuffle=True, num_workers=2, drop_last=True)

        self.run_dir = cfg.get("run_dir", "runs/jump_nce")
        os.makedirs(self.run_dir, exist_ok=True)

        self.p_jump = float(cfg.get("p_jump", 0.01))
        self.amp = float(cfg.get("amp", 3.0))
        self.lambda_eps = float(cfg.get("lambda_eps", 0.0))  # auxiliary eps loss weight

    def _sample_t(self, B):
        return torch.rand(B, device=self.device).clamp(1e-4, 1 - 1e-4)

    def train(self, epochs=10):
        bce = torch.nn.BCEWithLogitsLoss()

        self.model.train()
        for ep in range(epochs):
            pbar = tqdm.tqdm(self.dl, desc=f"[JUMP-NCE] Epoch {ep+1}/{epochs}")
            for (x0, _) in pbar:
                x0 = x0.to(self.device)                # (B,1,L)
                B = x0.size(0)
                t = self._sample_t(B)                  # (B,)
                lam = self.sch.log_snr(t)

                # sample positives from forward kernel
                xt_pos = sample_forward_xt(self.sch, x0, t)
                # negatives by impulsive jumps
                xt_neg = jump_corrupt(xt_pos, p=self.p_jump, amp=self.amp)

                eps_pos, logit_pos = self.model(xt_pos, lam)
                eps_neg, logit_neg = self.model(xt_neg, lam)

                # NCE classification (positives=1, negatives=0)
                y_pos = torch.ones_like(logit_pos)
                y_neg = torch.zeros_like(logit_neg)
                loss_cls = bce(logit_pos, y_pos) + bce(logit_neg, y_neg)

                # optional auxiliary eps-DSM on positives
                loss_eps = 0.0
                if self.lambda_eps > 0:
                    a = self.sch.alpha_bar(t).view(-1, 1, 1)
                    s = self.sch.sigma(t).view(-1, 1, 1)
                    eps_true = (xt_pos - a * x0) / (s + 1e-12)
                    loss_eps = torch.mean((eps_pos - eps_true) ** 2)

                loss = loss_cls + self.lambda_eps * loss_eps

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.ema.update(self.model)

                pbar.set_postfix(loss=float(loss.item()))

            save_ckpt(self.model, self.ema, self.opt, os.path.join(self.run_dir, "latest.pt"))
