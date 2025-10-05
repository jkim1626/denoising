import torch

def tweedie_plug_in(x, alpha_bar, sigma, score_pred):
    """E[X0|Xt=x] = (x + sigma^2 s)/alpha_bar  (additive VP)"""
    return (x + sigma**2 * score_pred) / (alpha_bar + 1e-12)

def denoiser_to_score(denoised, x, sigma):
    """Map denoiser output -> score via (D(x)-x)/sigma^2."""
    return (denoised - x) / (sigma**2 + 1e-12)
