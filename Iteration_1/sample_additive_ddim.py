# sample_additive_ddim.py
import torch, numpy as np
from torch.utils.data import DataLoader
from dataset_additive import AdditiveCleanDataset
from sde_schedule import VPDiffusionSchedule
from score_mlp import TinyScore1D

def psnr(x, y):
    mse = np.mean((x - y)**2)
    if mse == 0: return 99.0
    maxv = max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0)
    return 20*np.log10(maxv / np.sqrt(mse))

@torch.no_grad()
def ddim_denoise(x_T, model, sched, steps=50):
    """
    Deterministic DDIM/ODE sampling (eta=0) using eps-predicting model.
    x_T: [B,1,T] drawn from N(0, I), then progressively guided to x_0.
    """
    device = x_T.device
    N = sched.alpha_bars.numel()
    # pick `steps` indices from [N-1..0]
    ts = torch.linspace(N-1, 0, steps, device=device).long()
    x = x_T
    for idx in range(len(ts)-1):
        t_i   = ts[idx]
        t_im1 = ts[idx+1]
        a_bar_i   = sched.alpha_bars[t_i].view(1,1,1)
        a_bar_im1 = sched.alpha_bars[t_im1].view(1,1,1)
        sqrt_ab_i    = torch.sqrt(a_bar_i)
        sqrt_1mab_i  = torch.sqrt(1.0 - a_bar_i)

        t_scalar01 = (t_i.float() / (N-1)).expand(x.size(0))
        eps_hat = model(x, t_scalar01)

        # predicted x0 at step i
        x0_hat = (x - sqrt_1mab_i * eps_hat) / (sqrt_ab_i + 1e-8)

        # deterministic DDIM update to the next step
        sqrt_ab_next   = torch.sqrt(a_bar_im1)
        sqrt_1mab_next = torch.sqrt(1.0 - a_bar_im1)
        x = sqrt_ab_next * x0_hat + sqrt_1mab_next * eps_hat
    return x  # this is x_0 estimate

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_steps = 1000
    steps = 100  # try 20/50/100 to see tradeoffs

    ds = AdditiveCleanDataset("synthetic_data/additive")
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    sched = VPDiffusionSchedule(num_steps=num_steps, device=device)
    model = TinyScore1D().to(device)
    model.load_state_dict(torch.load("checkpoints/additive_eps.pt", map_location=device))
    model.eval()

    # Start from the *dataset noisy* or from pure noise?
    # For denoising, we want to start near the data’s noisy distribution.
    # Easiest: map a real x0 to x_T with a large t, then run DDIM back.
    psnrs = []
    for x0 in loader:
        x0 = x0.unsqueeze(1).to(device)  # [B,1,T]

        # pick a high noise level to mimic a corrupted observation
        N = sched.alpha_bars.numel()
        t_hi = int(0.9*(N-1))
        ab   = sched.alpha_bars[t_hi].view(1,1,1)
        x_T  = torch.sqrt(ab)*x0 + torch.sqrt(1.0 - ab)*torch.randn_like(x0)

        x0_hat = ddim_denoise(x_T, model, sched, steps=steps)

        x0_np = x0.squeeze(1).cpu().numpy()
        xh_np = x0_hat.squeeze(1).cpu().numpy()
        for i in range(x0_np.shape[0]):
            psnrs.append(psnr(x0_np[i], xh_np[i]))
        break  # one batch is enough to sanity check

    print(f"DDIM ({steps} steps) PSNR -> mean={np.mean(psnrs):.2f} dB, std={np.std(psnrs):.2f} dB")
