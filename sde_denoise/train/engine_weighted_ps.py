import os
import torch
import tqdm
from torch.utils.data import DataLoader

from ..utils.ema import EMA
from ..utils.ckpt import save_ckpt
from ..utils.logging import get_logger
from ..sde.schedule import CosineSchedule
from ..sde.additive_vp import AdditiveVP


def _per_sample_dsm_additive(model, sch, x0, t):
    """
    Per-sample DSM loss for additive VP.
      x_t = a x0 + s ε
      ε_target = (x_t - a x0) / s
      ε_pred = model(x_t, log_snr(t))
    Returns:
      per_loss: (B,) tensor of MSE over (C,L) per item
    """
    a = sch.alpha_bar(t).view(-1, 1, 1)
    s = sch.sigma(t).view(-1, 1, 1)
    eps = torch.randn_like(x0)
    xt = a * x0 + s * eps

    eps_tgt = (xt - a * x0) / (s + 1e-12)
    eps_prd = model(xt, sch.log_snr(t))
    per_loss = torch.mean((eps_prd - eps_tgt) ** 2, dim=(1, 2))  # (B,)
    return per_loss


def _loss_weight(t, sch, mode="logsnr_gamma", gamma=1.5):
    """
    Per-sample weight w(t). Normalized so E[w]=1 to keep LR comparable.
      mode = "none" | "logsnr_gamma" | "beta_gamma"
    """
    if mode == "none":
        w = torch.ones_like(t)
    elif mode == "logsnr_gamma":
        w = (sch.log_snr(t).abs() + 1e-12) ** gamma
    elif mode == "beta_gamma":
        w = (sch.beta(t) + 1e-12) ** gamma
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    return w / (w.mean() + 1e-12)


class TrainerWeightedPerSample:
    """
    Trainer that applies TRUE per-sample time weighting to the DSM loss.
    (Additive VP case.)
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

        self.run_dir = cfg.get("run_dir", "runs/additive_weighted_ps")
        os.makedirs(self.run_dir, exist_ok=True)

        self.weight_mode = cfg.get("time_weight_mode", "logsnr_gamma")
        self.gamma = float(cfg.get("time_weight_gamma", 1.5))

    def _sample_t(self, B):
        # keep away from the exact endpoints
        return torch.rand(B, device=self.device).clamp(1e-4, 1 - 1e-4)

    def train(self, epochs=10):
        self.model.train()
        for ep in range(epochs):
            pbar = tqdm.tqdm(self.dl, desc=f"[WPS] Epoch {ep+1}/{epochs}")
            for (x0, _) in pbar:
                x0 = x0.to(self.device)  # (B,1,L)
                B = x0.size(0)
                t = self._sample_t(B)    # (B,)

                # per-sample DSM and weights
                per_loss = _per_sample_dsm_additive(self.model, self.sch, x0, t)  # (B,)
                w = _loss_weight(t, self.sch, mode=self.weight_mode, gamma=self.gamma)  # (B,)

                loss = torch.mean(w * per_loss)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.ema.update(self.model)

                pbar.set_postfix(loss=float(loss.item()), w_mean=float(w.mean().item()))

            # save EMA checkpoint each epoch
            ck = os.path.join(self.run_dir, "latest.pt")
            save_ckpt(self.model, self.ema, self.opt, ck)
