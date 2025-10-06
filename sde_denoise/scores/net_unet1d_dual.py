import torch
import torch.nn as nn
import math

# Try to import the project's time embedding; fall back to a local compatible version.
try:
    from .time_embed import TimeFourier  # your repo's implementation
except Exception:
    class TimeFourier(nn.Module):
        """
        Fallback TimeFourier embedding compatible with TinyUNet1D expectations.
        Produces a (B, t_dim) embedding from a (B,) log-SNR vector.
        """
        def __init__(self, t_dim: int = 128, max_freq: float = 30.0):
            super().__init__()
            assert t_dim % 2 == 0, "t_dim must be even"
            self.t_dim = t_dim
            half = t_dim // 2
            # Log-spaced frequencies from 1 to max_freq
            freqs = torch.logspace(math.log10(1.0), math.log10(max_freq), steps=half)
            self.register_buffer("freqs", freqs, persistent=False)
            self.mlp = nn.Sequential(
                nn.Linear(t_dim, t_dim),
                nn.GELU(),
                nn.Linear(t_dim, t_dim),
            )

        def forward(self, lambda_t: torch.Tensor) -> torch.Tensor:
            if lambda_t.dim() == 0:
                lambda_t = lambda_t[None]
            # lambda_t: (B,)
            lamb = lambda_t[:, None] * self.freqs[None, :]        # (B, half)
            emb = torch.cat([torch.sin(lamb), torch.cos(lamb)], dim=1)  # (B, t_dim)
            return self.mlp(emb)


class TinyUNet1D_Dual(nn.Module):
    """
    A tiny 1D U-Net with two heads:
      - head_eps:   epsilon prediction (as in standard DSM)
      - head_logit: scalar classifier logit per item for NCE density-ratio learning
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, t_dim: int = 128):
        super().__init__()
        self.temb = TimeFourier(t_dim=t_dim)
        ch = 64

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

        self.mid = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.up = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

        self.dec1 = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # simple time conditioning
        self.fc_t = nn.Sequential(
            nn.Linear(t_dim, ch),
            nn.GELU(),
            nn.Linear(ch, ch),
        )

        # heads
        self.head_eps = nn.Conv1d(ch, out_ch, kernel_size=3, padding=1)
        self.head_logit = nn.Conv1d(ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, lambda_t: torch.Tensor):
        """
        x: (B, 1, L)
        lambda_t: (B,) log-SNR embedding input
        returns:
          eps:   (B, 1, L)
          logit: (B,)  (global pooled)
        """
        te = self.temb(lambda_t)           # (B, t_dim)
        tfeat = self.fc_t(te)[:, :, None]  # (B, ch, 1)

        h = self.enc1(x)
        h = self.down(h)
        h = h + tfeat
        h = self.mid(h)
        h = self.up(h)
        h = self.dec1(h)

        eps = self.head_eps(h)                     # (B, 1, L)
        logit_map = self.head_logit(h)             # (B, 1, L)
        logit = logit_map.mean(dim=(1, 2))         # (B,)
        return eps, logit
