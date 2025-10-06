import os
import torch
import tqdm
from torch.utils.data import DataLoader

from ..utils.ema import EMA
from ..utils.ckpt import save_ckpt
from ..utils.logging import get_logger
from ..sde.schedule import CosineSchedule
from ..sde.geometric_vp import GeometricVP  # VP in log-domain (a.k.a. "multiplicative")


def _per_sample_dsm_geometric_log(model, sch, x0, t, eps_floor=1e-6):
    """
    Closed-form conditional DSM in LOG domain (multiplicative noise).
      y0 = log(|x0| + eps_floor)
      Forward kernel in log-space (VP):  y_t = y0 - Λ(t) + sqrt(Λ(t)) * ε,  ε~N(0,I)
      Target epsilon: ε_tgt = (y_t - (y0 - Λ)) / sqrt(Λ)
      Model head predicts ε; input is y_t, time embedding logSNR(t).
    Returns per-sample ε-MSE reduced over (C,L).
    """
    # log-domain embedding of the clean signal (robust to zeros)
    y0 = torch.log(torch.clamp(x0.abs(), min=eps_floor))

    Lam = sch.Lambda(t).view(-1, 1, 1)                 # Λ(t)
    std = torch.sqrt(torch.clamp(Lam, min=1e-12))      # √Λ

    eps = torch.randn_like(x0)
    y_t = y0 - Lam + std * eps

    eps_tgt = (y_t - (y0 - Lam)) / (std + 1e-12)       # closed-form conditional target
    eps_prd = model(y_t, sch.log_snr(t))

    per_loss = torch.mean((eps_prd - eps_tgt) ** 2, dim=(1, 2))  # (B,)
    return per_loss


class TrainerMultiplicativePerSample:
    """
    Per-sample DSM trainer in log domain for multiplicative (geometric VP) noise.
    """
    def __init__(self, cfg, dataset, model, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.logger = get_logger()

        self.sch = CosineSchedule(device=device)
        self.sde = GeometricVP(self.sch, device=device)  # used for sampling later if needed

        self.model = model.to(device)
        self.ema = EMA(self.model, decay=float(cfg.get("ema", 0.999)))
        self.opt = torch.optim.AdamW(self.model.parameters(),
                                     lr=float(cfg.get("lr", 2e-4)),
                                     weight_decay=1e-4)

        self.dl = DataLoader(dataset,
                             batch_size=int(cfg.get("batch", 16)),
                             shuffle=True, num_workers=2, drop_last=True)

        self.run_dir = cfg.get("run_dir", "runs/multiplicative_ps")
        os.makedirs(self.run_dir, exist_ok=True)

    def _sample_t(self, B):
        return torch.rand(B, device=self.device).clamp(1e-4, 1-1e-4)

    def train(self, epochs=10):
        self.model.train()
        for ep in range(epochs):
            pbar = tqdm.tqdm(self.dl, desc=f"[MUL] Epoch {ep+1}/{epochs}")
            for (x0, _) in pbar:
                x0 = x0.to(self.device)     # (B,1,L), can contain zeros/positives
                B = x0.size(0)
                t = self._sample_t(B)

                per = _per_sample_dsm_geometric_log(self.model, self.sch, x0, t)
                loss = per.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.ema.update(self.model)
                pbar.set_postfix(loss=float(loss.item()))

            ck = os.path.join(self.run_dir, "latest.pt")
            save_ckpt(self.model, self.ema, self.opt, ck)
