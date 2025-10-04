# score_mlp.py
import torch
import torch.nn as nn
import math

def sinusoidal_time_emb(t, dim):
    """
    t: [B] in [0, 1] or [0..N-1]/N. returns [B, dim]
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class TinyScore1D(nn.Module):
    """
    Very small ConvNet: input [B,1,T], predicts ε̂ same shape.
    Time embedding is broadcast as extra channels.
    """
    def __init__(self, time_dim=64, base_ch=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )
        self.in_conv = nn.Conv1d(1 + base_ch, base_ch, kernel_size=5, padding=2)
        self.block1 = nn.Sequential(nn.Conv1d(base_ch, base_ch, 5, padding=2), nn.SiLU())
        self.block2 = nn.Sequential(nn.Conv1d(base_ch, base_ch, 5, padding=2), nn.SiLU())
        self.out_conv = nn.Conv1d(base_ch, 1, kernel_size=5, padding=2)

    def forward(self, x, t_scalar01):
        """
        x: [B,1,T], t_scalar01: [B] in [0,1]
        """
        t_emb = sinusoidal_time_emb(t_scalar01, self.time_dim)   # [B,time_dim]
        t_feat = self.time_mlp(t_emb)                            # [B,C]
        # Broadcast time features along T and concat as channels
        B, _, T = x.shape
        t_ch = t_feat[:, :, None].expand(B, t_feat.shape[1], T)
        h = torch.cat([x, t_ch], dim=1)                          # [B, 1+C, T]
        h = self.in_conv(h)
        h = self.block1(h) + h
        h = self.block2(h) + h
        eps_hat = self.out_conv(h)
        return eps_hat
