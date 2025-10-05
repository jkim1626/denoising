import torch

def dsm_additive(eps_pred, x_t, x0, alpha_bar, sigma):
    # x_t = alpha_bar x0 + sigma ε  ⇒ target ε
    eps = (x_t - alpha_bar * x0) / (sigma + 1e-12)
    return torch.mean((eps_pred - eps)**2)

def dsm_geometric_logspace(eps_pred, y_t, y0, Lambda_t):
    # y_t = y0 - Λ_t + sqrt(Λ_t) ε  ⇒ target ε
    eps = (y_t - (y0 - Lambda_t)) / torch.sqrt(Lambda_t + 1e-12)
    return torch.mean((eps_pred - eps)**2)
