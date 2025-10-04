# sde_schedule.py
import torch
import math

class VPDiffusionSchedule:
    """
    DDPM/VP schedule for additive Gaussian noise.
    We train the model to predict ε (the noise) at a random discrete step t.
    Score can be recovered as s(x_t,t) = -ε / sigma_t, where sigma_t = sqrt(1 - alpha_bar_t).
    """
    def __init__(self, num_steps: int = 1000, device="cpu", s: float = 0.008):
        import torch, math
        self.device = torch.device(device)
        self.num_steps = num_steps
        t = torch.linspace(0.0, 1.0, num_steps, device=self.device)
        f = lambda u: torch.cos(((u + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bars = f(t) / f(torch.tensor(0.0, device=self.device))  # normalize so ā(0)=1
        alphas = torch.empty_like(alpha_bars)
        alphas[0] = alpha_bars[0]
        alphas[1:] = alpha_bars[1:] / alpha_bars[:-1].clamp_min(1e-12)
        betas = (1.0 - alphas).clamp(1e-6, 0.999)
        self.alpha_bars, self.alphas, self.betas = alpha_bars, alphas, betas

    @torch.no_grad()
    def sample_xt(self, x0: torch.Tensor, t_index: torch.Tensor):
        """
        x0: [B, 1, T]
        t_index: [B] integer indices in [0, num_steps-1]
        returns: x_t, eps, sqrt_alpha_bar, sqrt_one_minus_alpha_bar
        """
        ab = self.alpha_bars[t_index].view(-1, 1, 1)
        sqrt_ab = torch.sqrt(ab)
        sqrt_1mab = torch.sqrt(1.0 - ab)
        eps = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_1mab * eps
        return x_t, eps, sqrt_ab, sqrt_1mab
