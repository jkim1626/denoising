import torch
from .base import SDE
from .schedule import CosineSchedule

class GeometricVP(SDE):
    def __init__(self, schedule: CosineSchedule, **kw):
        super().__init__(**kw)
        self.sch = schedule

    def drift(self, x, t):
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        return -0.5 * b * x

    def diffusion_linop(self, x, t):
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        return lambda xi: torch.sqrt(b) * (x * xi)

    def a_matrix(self, x, t):
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        d = x.shape[-1]
        eye = torch.eye(d, device=x.device).expand(x.size(0), d, d)
        # a = 1/2 b * diag(x)^2  (represented as diag via outer with eye)
        # matvec(a, s) will be elementwise: 0.5*b * x^2 * s
        return 0.5 * b.view(-1, 1, 1) * eye * (x[:,0,:]**2).unsqueeze(-1)

    def div_a(self, x, t):
        # ∇·a = b * x  (1D diag case)
        b = self.sch.beta(t).view(-1, *([1]*(x.ndim-1)))
        return b * x
