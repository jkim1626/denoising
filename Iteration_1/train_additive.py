# train_additive.py
import os, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_additive import AdditiveCleanDataset
from sde_schedule import VPDiffusionSchedule
from score_mlp import TinyScore1D

# ----------------- config -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32          # lower if you hit OOM
num_steps = 1000         # diffusion steps (must match sde_schedule)
epochs = 30              # train longer for better PSNR
lr = 1e-3
ema_decay = 0.999

# ----------------- data -------------------
train_root = "synthetic_data/additive"   # your folder with .npz files (no splits)
train_ds = AdditiveCleanDataset(train_root)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

# ------------- schedule & model -----------
sched = VPDiffusionSchedule(num_steps=num_steps, device=device)  # cosine schedule if you updated sde_schedule.py
model = TinyScore1D(time_dim=64, base_ch=64).to(device)
ema_model = copy.deepcopy(model).to(device)
for p in ema_model.parameters():
    p.requires_grad = False

opt = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction="none")  # we'll weight per-sample by sigma^2(t)

# ----------------- train ------------------
for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
    for x0 in pbar:
        # x0: [B, T] -> [B, 1, T]
        x0 = x0.unsqueeze(1).to(device)

        # sample random diffusion step per item
        B = x0.size(0)
        t_index = torch.randint(0, num_steps, (B,), device=device)   # [B]
        t_scalar01 = t_index.float() / (num_steps - 1)               # [B]

        # forward sample x_t = sqrt(ā_t)*x0 + sqrt(1-ā_t)*ε
        x_t, eps, _, _ = sched.sample_xt(x0, t_index)

        # predict ε̂
        eps_hat = model(x_t, t_scalar01)

        # weighted MSE: w(t) = sigma^2(t) = 1 - ā_t
        with torch.no_grad():
            sigma2 = (1.0 - sched.alpha_bars[t_index]).view(-1, 1, 1)  # [B,1,1]
        mse = criterion(eps_hat, eps)                                   # [B,1,T]
        loss = (sigma2 * mse).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # EMA update
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        pbar.set_postfix(loss=float(loss.item()))

# --------------- save ckpts ---------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/additive_eps.pt")
torch.save(ema_model.state_dict(), "checkpoints/additive_eps_ema.pt")
print("Saved -> checkpoints/additive_eps.pt and checkpoints/additive_eps_ema.pt")
