# Iteration 2

import torch

def loss_weight(t, sch, mode="logsnr_gamma", gamma=1.5):
    """
    Returns a per-sample scalar weight w(t).
    Modes:
      - "none"            : 1
      - "logsnr_gamma"    : |logSNR(t)|^gamma normalized by mean (stable)
      - "beta_gamma"      : beta(t)^gamma normalized by mean
    """
    if mode == "none":
        return torch.ones_like(t)
    elif mode == "logsnr_gamma":
        ls = sch.log_snr(t).abs()
        w = (ls + 1e-12) ** gamma
    elif mode == "beta_gamma":
        b = sch.beta(t)
        w = (b + 1e-12) ** gamma
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    # Normalize to mean 1.0 to keep effective LR comparable
    return w / (w.mean() + 1e-12)
