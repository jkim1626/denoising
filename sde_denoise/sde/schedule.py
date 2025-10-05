import math, torch

class CosineSchedule:
    """Cosine noise schedule utilities on t∈[0,1]."""
    def __init__(self, beta_max: float = 20.0, device="cuda"):
        self.beta_max = beta_max
        self.device = device
        # fixed grid for numerical Λ(t) integration
        self._grid = torch.linspace(0.0, 1.0, 2049, device=device)  # N=2049 → Δ=1/(N-1)
        self._beta_g = self._beta(self._grid)
        self._Lam_g  = torch.cumsum(self._beta_g, dim=0) * (1.0 / (len(self._grid) - 1))

    def _beta(self, t):
        return (torch.sin(0.5*math.pi*t)**2) * self.beta_max

    def beta(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device)
        return self._beta(t)

    def _interp1d(self, y, x):
        """
        Piecewise-linear interpolation of y(grid) at x ∈ [0,1].
        Safe at boundaries: clamps x to [0, 1 - tiny], so i1 is always valid.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=y.device)
        x = x.to(y.device)
        # keep strictly inside the last cell
        x = x.clamp(0.0, 1.0 - 1e-8)
        n = len(self._grid)  # 2049
        idx_f = x * (n - 1)  # in [0, n-1)
        i0 = idx_f.floor().long()          # in [0, n-2]
        i1 = (i0 + 1).clamp(max=n - 1)     # in [1, n-1]
        w  = (idx_f - i0.float())
        return (1 - w) * y[i0] + w * y[i1]

    def Lambda(self, t):
        return self._interp1d(self._Lam_g, t)

    def alpha_bar(self, t):
        Lam = self.Lambda(t)
        return torch.exp(-0.5 * Lam)

    def sigma(self, t):
        Lam = self.Lambda(t)
        return torch.sqrt(1.0 - torch.exp(-Lam)).clamp(min=1e-12)

    def log_snr(self, t):
        a = self.alpha_bar(t)
        s = self.sigma(t)
        return torch.log(a / s)
