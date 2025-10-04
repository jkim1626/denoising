# eval_additive_one_step.py
import os, torch, numpy as np
from torch.utils.data import DataLoader
from dataset_additive import AdditiveCleanDataset
from sde_schedule import VPDiffusionSchedule
from score_mlp import TinyScore1D

def psnr(x, y):
    mse = np.mean((x - y)**2)
    if mse == 0: return 99.0
    maxv = max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0)
    return 20*np.log10(maxv / np.sqrt(mse))

device = "cuda" if torch.cuda.is_available() else "cpu"
num_steps = 1000

ds = AdditiveCleanDataset("synthetic_data/additive")
loader = DataLoader(ds, batch_size=8, shuffle=False)

sched = VPDiffusionSchedule(num_steps=num_steps, device=device)
model = TinyScore1D(time_dim=64, base_ch=64).to(device)

# prefer EMA if it exists
ckpt_path = "checkpoints/additive_eps_ema.pt" if os.path.exists("checkpoints/additive_eps_ema.pt") else "checkpoints/additive_eps.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

psnrs = []
with torch.no_grad():
    for x0 in loader:
        x0 = x0.unsqueeze(1).to(device)
        B = x0.size(0)
        t_index = torch.full((B,), int(0.6*(num_steps-1)), device=device, dtype=torch.long)
        t_scalar01 = t_index.float() / (num_steps-1)

        x_t, eps, sqrt_ab, sqrt_1mab = sched.sample_xt(x0, t_index)
        eps_hat = model(x_t, t_scalar01)
        x0_hat = (x_t - sqrt_1mab * eps_hat) / (sqrt_ab + 1e-8)

        x0_np = x0.squeeze(1).cpu().numpy()
        xh_np = x0_hat.squeeze(1).cpu().numpy()
        for i in range(B):
            psnrs.append(psnr(x0_np[i], xh_np[i]))
        break  # just check one batch

print(f"EMA one-step x0-estimate PSNR -> mean={np.mean(psnrs):.2f} dB, std={np.std(psnrs):.2f} dB")
