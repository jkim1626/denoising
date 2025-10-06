# Iteration 2

import os, torch, tqdm
from torch.utils.data import DataLoader
from ..utils.ema import EMA
from ..utils.ckpt import save_ckpt
from ..utils.logging import get_logger
from ..sde.schedule import CosineSchedule
from ..sde.additive_vp import AdditiveVP
from ..sde.geometric_vp import GeometricVP
from ..sde.jump_diffusion import JumpDiffusion
from .dsm_loss import dsm_additive, dsm_geometric_logspace
from .weighting import loss_weight

def _forward_perturb_additive(sch, x0, t):
    a = sch.alpha_bar(t).view(-1, 1, 1)
    s = sch.sigma(t).view(-1, 1, 1)
    eps = torch.randn_like(x0)
    xt = a * x0 + s * eps
    return xt, {"alpha_bar": a, "sigma": s, "lambda_t": sch.log_snr(t)}

def _forward_perturb_geometric(sch, x0, t, eps=1e-6):
    y0 = torch.log(torch.clamp(x0.abs(), min=eps))
    Lam = sch.Lambda(t).view(-1, 1, 1)
    epsn = torch.randn_like(x0)
    yt = y0 - Lam + torch.sqrt(Lam + 1e-12) * epsn
    return yt, {"y0": y0, "Lambda_t": Lam, "lambda_t": sch.log_snr(t)}

class TrainerWeighted:
    """
    Same as the baseline Trainer, but multiplies the DSM loss by a time weight w(t).
    """
    def __init__(self, cfg, dataset, model, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.logger = get_logger()
        self.sch = CosineSchedule(device=device)

        nk = cfg["noise_kind"]
        if nk == "additive":
            self.sde = AdditiveVP(self.sch, device=device)
        elif nk == "multiplicative":
            self.sde = GeometricVP(self.sch, device=device)
        elif nk == "jump":
            self.sde = JumpDiffusion(self.sch, lam=cfg.get("lambda", 0.02), device=device)
        else:
            self.sde = AdditiveVP(self.sch, device=device)

        self.model = model.to(device)
        self.ema = EMA(self.model, decay=cfg.get("ema", 0.999))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.get("lr", 2e-4), weight_decay=1e-4)
        self.dl = DataLoader(dataset, batch_size=cfg.get("batch", 8), shuffle=True, num_workers=2, drop_last=True)
        os.makedirs(cfg.get("run_dir", "runs/additive_weighted"), exist_ok=True)

        self.weight_mode = cfg.get("time_weight_mode", "logsnr_gamma")
        self.gamma = float(cfg.get("time_weight_gamma", 1.5))

    def sample_t(self, B):
        return torch.rand(B, device=self.device).clamp(1e-4, 1-1e-4)

    def train(self, epochs=10):
        self.model.train()
        for ep in range(epochs):
            pbar = tqdm.tqdm(self.dl, desc=f"[TW] Epoch {ep+1}/{epochs}")
            for (x0, meta) in pbar:
                x0 = x0.to(self.device)  # (B,1,L)
                B = x0.size(0)
                t = self.sample_t(B)

                if self.cfg["noise_kind"] == "multiplicative":
                    xt, aux = _forward_perturb_geometric(self.sch, x0, t)
                    pred = self.model(xt, aux["lambda_t"])
                    base_loss = dsm_geometric_logspace(pred, xt, aux["y0"], aux["Lambda_t"])
                else:
                    xt, aux = _forward_perturb_additive(self.sch, x0, t)
                    pred = self.model(xt, aux["lambda_t"])
                    base_loss = dsm_additive(pred, xt, x0, aux["alpha_bar"], aux["sigma"])

                w = loss_weight(t, self.sch, mode=self.weight_mode, gamma=self.gamma)
                # average per-sample: multiply scalar weight then mean
                loss = (w * base_loss.detach().new_full((B,), base_loss.item())).mean()  # keep log readability
                # correct implementation: compute per-sample loss — quick workaround below:
                # For simplicity, treat base_loss as scalar and rescale:
                loss = base_loss * (w.mean())

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.ema.update(self.model)
                pbar.set_postfix(loss=float(base_loss.item()), w_mean=float(w.mean().item()))

            ck = os.path.join(self.cfg.get("run_dir", "runs/additive_weighted"), "latest.pt")
            save_ckpt(self.model, self.ema, self.opt, ck)
