# train_additive_unet.py
import os, copy, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_additive import AdditiveCleanDataset
from sde_schedule import VPDiffusionSchedule

# ---------------- time embedding ----------------
def sinusoidal_time_emb(t, dim):
    # t: [B] in [0,1], returns [B, dim]
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# ---------------- U-Net blocks -------------------
class ResBlock1D(nn.Module):
    def __init__(self, ch, tdim, kernel=5):
        super().__init__()
        pad = kernel // 2
        self.norm1 = nn.GroupNorm(8, ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad)
        self.time  = nn.Sequential(nn.SiLU(), nn.Linear(tdim, ch))
        self.norm2 = nn.GroupNorm(8, ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad)

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        # add time embedding (broadcast)
        te = self.time(t_emb)[:, :, None]
        h = h + te
        h = self.conv2(self.act2(self.norm2(h)))
        return x + h

class Down(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size=3, stride=2, padding=1)
        self.rb1  = ResBlock1D(cout, tdim)
        self.rb2  = ResBlock1D(cout, tdim)

    def forward(self, x, t_emb):
        x = self.conv(x)
        x = self.rb1(x, t_emb)
        x = self.rb2(x, t_emb)
        return x

class Up(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1)
        self.rb1    = ResBlock1D(cout, tdim)
        self.rb2    = ResBlock1D(cout, tdim)

    def forward(self, x, t_emb):
        x = self.deconv(x)
        x = self.rb1(x, t_emb)
        x = self.rb2(x, t_emb)
        return x

class UNet1D(nn.Module):
    """
    Small 1D U-Net for epsilon prediction: input [B,1,T] -> output [B,1,T]
    """
    def __init__(self, base=64, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, tdim), nn.SiLU(),
            nn.Linear(tdim, tdim)
        )

        self.in_conv = nn.Conv1d(1, base, kernel_size=5, padding=2)

        self.down1 = Down(base,      base,      tdim)
        self.down2 = Down(base,      base*2,    tdim)
        self.down3 = Down(base*2,    base*2,    tdim)

        self.mid1  = ResBlock1D(base*2, tdim)
        self.mid2  = ResBlock1D(base*2, tdim)

        self.up3   = Up(base*2,    base*2,   tdim)
        self.up2   = Up(base*2,    base,     tdim)
        self.up1   = Up(base,      base,     tdim)

        self.out   = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv1d(base, 1, kernel_size=5, padding=2)
        )

    def forward(self, x, t01):
        # t01: [B] in [0,1]
        t_emb = sinusoidal_time_emb(t01, self.tdim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)     # [B, base, T/2]
        d2 = self.down2(d1, t_emb)     # [B, 2*base, T/4]
        d3 = self.down3(d2, t_emb)     # [B, 2*base, T/8]

        m  = self.mid2(self.mid1(d3, t_emb), t_emb)

        u3 = self.up3(m,  t_emb)
        # center-crop/align if shapes mismatch due to odd lengths
        if u3.shape[-1] != d2.shape[-1]:
            minT = min(u3.shape[-1], d2.shape[-1])
            u3 = u3[..., :minT]; d2 = d2[..., :minT]
        u3 = u3 + d2

        u2 = self.up2(u3, t_emb)
        if u2.shape[-1] != d1.shape[-1]:
            minT = min(u2.shape[-1], d1.shape[-1])
            u2 = u2[..., :minT]; d1 = d1[..., :minT]
        u2 = u2 + d1

        u1 = self.up1(u2, t_emb)
        if u1.shape[-1] != x0.shape[-1]:
            minT = min(u1.shape[-1], x0.shape[-1])
            u1 = u1[..., :minT]; x0 = x0[..., :minT]
        u1 = u1 + x0

        out = self.out(u1)
        return out

# ---------------- config -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16          # UNet is bigger; adjust if OOM
num_steps = 1000
epochs = 25
lr = 1e-3
ema_decay = 0.999

# ---------------- data ---------------------
train_ds = AdditiveCleanDataset("synthetic_data/additive")
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

# ------------- schedule --------------------
# expects you already switched to cosine schedule in sde_schedule.py
sched = VPDiffusionSchedule(num_steps=num_steps, device=device)

# ------------- model & EMA -----------------
model = UNet1D(base=64, tdim=128).to(device)
ema_model = copy.deepcopy(model).to(device)
for p in ema_model.parameters(): p.requires_grad = False

opt = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction="none")

# ---------------- train --------------------
for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
    for x0 in pbar:
        x0 = x0.unsqueeze(1).to(device)  # [B,1,T]
        B = x0.size(0)
        t_index = torch.randint(0, num_steps, (B,), device=device)
        t01     = t_index.float() / (num_steps - 1)

        x_t, eps, _, _ = sched.sample_xt(x0, t_index)
        eps_hat = model(x_t, t01)

        with torch.no_grad():
            sigma2 = (1.0 - sched.alpha_bars[t_index]).view(-1,1,1)
        mse  = criterion(eps_hat, eps)
        loss = (sigma2 * mse).mean()

        opt.zero_grad(); loss.backward(); opt.step()

        # EMA update
        with torch.no_grad():
            for q, p in zip(ema_model.parameters(), model.parameters()):
                q.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        pbar.set_postfix(loss=float(loss.item()))

# --------------- save ckpts ----------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(),     "checkpoints/additive_unet_eps.pt")
torch.save(ema_model.state_dict(), "checkpoints/additive_unet_eps_ema.pt")
print("Saved -> checkpoints/additive_unet_eps.pt and checkpoints/additive_unet_eps_ema.pt")
