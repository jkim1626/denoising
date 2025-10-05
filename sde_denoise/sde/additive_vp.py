import torch
from .base import SDE
from .schedule import CosineSchedule

class AdditiveVP(SDE):
    def __init__(self, schedule: CosineSchedule, **kw):
        super().__init__(**kw)
        self.sch = schedule

    def drift(self, x, t):
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        return -0.5 * b * x

    def diffusion_linop(self, x, t):
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        return lambda xi: torch.sqrt(b) * xi

    def a_matrix(self, x, t):
        # 1D spectra -> diagonal scalar per sample/channel/position
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        # treat a as scalar acting elementwise; implement with identity via matvec (broadcast)
        d = x.shape[-1]
        eye = torch.eye(d, device=x.device).expand(x.size(0), d, d)
        return 0.5 * b.view(-1, 1, 1) * eye
