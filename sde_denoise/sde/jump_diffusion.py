import torch
from .base import SDE
from .schedule import CosineSchedule

class JumpDiffusion(SDE):
    def __init__(self, schedule: CosineSchedule, lam: float = 0.02, **kw):
        super().__init__(**kw)
        self.sch = schedule
        self.lam = lam  # Poisson rate per unit time

    def drift(self, x, t):
        return torch.zeros_like(x)

    def diffusion_linop(self, x, t):
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        return lambda xi: torch.sqrt(b) * xi

    def a_matrix(self, x, t):
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        d = x.shape[-1]
        eye = torch.eye(d, device=x.device).expand(x.size(0), d, d)
        return 0.5 * b.view(-1, 1, 1) * eye

    def sample_num_jumps(self, shape, dt, device):
        rate = self.lam * dt
        return torch.poisson(torch.full(shape, rate, device=device)).long()
