# eval_from_dataset_noisy.py
import glob, os, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from sde_schedule import VPDiffusionSchedule
from score_mlp import TinyScore1D

def psnr(x, y):
    mse = np.mean((x - y)**2)
    if mse == 0: return 99.0
    maxv = max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0)
    return 20*np.log10(maxv / np.sqrt(mse))

class AdditivePairDataset(Dataset):
    """Loads (clean, noisy) pairs from your existing additive folder (no splits)."""
    def __init__(self, root="synthetic_data/additive"):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {root}")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        z = np.load(self.files[idx])
        clean = z["clean"].astype(np.float32)
        noisy = z["noisy"].astype(np.float32)
        return torch.from_numpy(clean), torch.from_numpy(noisy)

@torch.no_grad()
def ddim_back_from_t(observed_xt, t_index, model, sched, steps=50):
    """DDIM (eta=0) integrate from given t_index down to 0."""
    device = observed_xt.device
    N = sched.alpha_bars.numel()
    ts = torch.linspace(int(t_index), 0, steps, device=device).long()
    x = observed_xt
    for k in range(len(ts)-1):
        t_i   = ts[k]
        t_im1 = ts[k+1]
        a_bar_i   = sched.alpha_bars[t_i].view(1,1,1)
        a_bar_im1 = sched.alpha_bars[t_im1].view(1,1,1)
        sqrt_ab_i    = torch.sqrt(a_bar_i)
        sqrt_1mab_i  = torch.sqrt(1.0 - a_bar_i)
        t_scalar01 = (t_i.float() / (N-1)).expand(x.size(0))
        eps_hat = model(x, t_scalar01)
        x0_hat = (x - sqrt_1mab_i * eps_hat) / (sqrt_ab_i + 1e-8)
        sqrt_ab_next   = torch.sqrt(a_bar_im1)
        sqrt_1mab_next = torch.sqrt(1.0 - a_bar_im1)
        x = sqrt_ab_next * x0_hat + sqrt_1mab_next * eps_hat
    return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_steps = 1000
    steps = 50

    ds = AdditivePairDataset("synthetic_data/additive")
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    sched = VPDiffusionSchedule(num_steps=num_steps, device=device)
    model = TinyScore1D().to(device)
    # If you have EMA (from step 15), prefer loading it:
    ckpt_path = "checkpoints/additive_eps_ema.pt" if os.path.exists("checkpoints/additive_eps_ema.pt") else "checkpoints/additive_eps.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    psnrs = []
    for clean, noisy in loader:
        clean = clean.unsqueeze(1).to(device)   # [B,1,T]
        noisy = noisy.unsqueeze(1).to(device)   # [B,1,T]

        # estimate per-sample a_bar from variance ratio r ≈ 1 - a_bar
        noise_res = noisy - clean
        var_clean = torch.var(clean, dim=2, unbiased=False, keepdim=True)
        var_noise = torch.var(noise_res, dim=2, unbiased=False, keepdim=True)
        r = (var_noise / (var_noise + var_clean + 1e-12)).clamp(0.0, 0.999)     # [B,1,1]
        a_bar_est = (1.0 - r).squeeze(-1).squeeze(-1)                           # [B]

        # nearest t index per sample
        diffs = (sched.alpha_bars[None, :] - a_bar_est[:, None]).abs()
        t_index = torch.argmin(diffs, dim=1)                                     # [B]

        # construct observed x_t consistent with additive mixture:
        # x_t_hat = sqrt(ā)*clean + (noisy - clean) = noisy + (sqrt(ā)-1)*clean
        sqrt_ab = torch.sqrt(a_bar_est)[:, None, None]
        observed_xt = noisy + (sqrt_ab - 1.0) * clean                            # [B,1,T]

        # ---- per-sample denoising from its own t ----
        x0_hats = []
        for i in range(clean.size(0)):
            x_i  = observed_xt[i:i+1]
            ti   = t_index[i].item()
            x0_i = ddim_back_from_t(x_i, ti, model, sched, steps=steps)
            x0_hats.append(x0_i)
        x0_hat = torch.cat(x0_hats, dim=0)                                       # [B,1,T]

        x0_np = clean.squeeze(1).cpu().numpy()
        xh_np = x0_hat.squeeze(1).cpu().numpy()
        for i in range(x0_np.shape[0]):
            psnrs.append(psnr(x0_np[i], xh_np[i]))
        break  # one batch is enough for a quick check

    print(f"Dataset-noisy DDIM ({steps} steps, per-sample t) PSNR -> mean={np.mean(psnrs):.2f} dB, std={np.std(psnrs):.2f} dB")
