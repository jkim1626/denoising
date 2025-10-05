import torch, torch.nn as nn
import einops as eo
from .time_embed import LogSNREmbed

def ConvBN(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv1d(in_c, out_c, k, stride=s, padding=p),
                         nn.GroupNorm(8, out_c), nn.SiLU())

class TinyUNet1D(nn.Module):
    """Compact 1D U-Net for spectra shaped (B, C=1, L)."""
    def __init__(self, in_ch=1, out_ch=1, t_dim=128, base=64):
        super().__init__()
        self.temb = LogSNREmbed(t_dim)
        C = base
        self.enc1 = ConvBN(in_ch + t_dim, C, 7, 1, 3)
        self.enc2 = ConvBN(C, 2*C, 4, 2, 1)
        self.mid1 = ConvBN(2*C, 2*C, 3, 1, 1)
        self.mid2 = ConvBN(2*C, 2*C, 3, 1, 1)
        self.up   = nn.ConvTranspose1d(2*C, C, 4, stride=2, padding=1)
        self.dec  = ConvBN(C, C, 3, 1, 1)
        self.out  = nn.Conv1d(C, out_ch, 1)

    def forward(self, x, lambda_t):
        # x: (B,1,L)
        B, C, L = x.shape
        te = self.temb(lambda_t)              # (B, t_dim)
        te_1d = te.unsqueeze(-1).repeat(1,1,L)
        h = torch.cat([x, te_1d], dim=1)      # (B, 1+t_dim, L)
        h1 = self.enc1(h)
        h2 = self.enc2(h1)
        m  = self.mid2(self.mid1(h2))
        u  = self.up(m)
        u  = self.dec(u)
        y  = self.out(u)
        return y
