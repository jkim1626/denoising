import os
import torch
import tqdm
from torch.utils.data import DataLoader

from ..utils.ema import EMA
from ..utils.ckpt import save_ckpt
from ..utils.logging import get_logger
from ..sde.schedule import CosineSchedule
from ..sde.additive_vp import AdditiveVP

def _per_sample_dsm_additive_logspace(model, sch, x0, t, eps_floor=1e-6):
    """
    Train in log-space: y0 = log(x0 + eps_floor)
    Forward VP in log-space: y_t = a y0 + s ε
    DSM target: ε_tgt = (y_t - a y0) / s
    """
    y0 = torch.log(torch.clamp(x0, min=eps_floor))
    a = sch.alpha_bar(t).view(-1,1,1)
    s = sch.sigma(t).view(-1,1,1)
    eps = torch.randn_like(y0)
    yt  = a * y0 + s * eps
    eps_tgt = (yt - a * y0) / (s + 1e-12)
    eps_prd = model(yt, sch.log_snr(t))
    per = torch.mean((eps_prd - eps_tgt)**2, dim=(1,2))
    return per

class TrainerLogVPPerSample:
    """
    Per-sample DSM trainer in LOG-space using the additive VP schedule.
    """
    def __init__(self, cfg, dataset, model, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.logger = get_logger()
        self.sch = CosineSchedule(device=device)
        self.sde = AdditiveVP(self.sch, device=device)  # sampling utilities if needed

        self.model = model.to(device)
        self.ema = EMA(self.model, decay=float(cfg.get("ema", 0.999)))
        self.opt = torch.optim.AdamW(self.model.parameters(),
                                     lr=float(cfg.get("lr", 2e-4)),
                                     weight_decay=1e-4)

        self.dl = DataLoader(dataset,
                             batch_size=int(cfg.get("batch", 16)),
                             shuffle=True, num_workers=2, drop_last=True)

        self.run_dir = cfg.get("run_dir", "runs/multiplicative_logvp_ps")
        os.makedirs(self.run_dir, exist_ok=True)

    def _sample_t(self, B):  # avoid endpoints
        return torch.rand(B, device=self.device).clamp(1e-4, 1-1e-4)

    def train(self, epochs=10):
        self.model.train()
        for ep in range(epochs):
            pbar = tqdm.tqdm(self.dl, desc=f"[LOGVP] Epoch {ep+1}/{epochs}")
            for (x0, _) in pbar:
                x0 = x0.to(self.device)              # (B,1,L); must be non-negative
                t  = self._sample_t(x0.size(0))
                per = _per_sample_dsm_additive_logspace(self.model, self.sch, x0, t)
                loss = per.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.ema.update(self.model)
                pbar.set_postfix(loss=float(loss.item()))
            save_ckpt(self.model, self.ema, self.opt, os.path.join(self.run_dir, "latest.pt"))
