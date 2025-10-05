import torch

def reverse_sde_step(sde, score_fn, x, t, dt):
    drift = sde.reverse_drift(x, t, score_fn)
    g = sde.diffusion_linop(x, t)
    noise = torch.randn_like(x)
    return x + drift*dt + g(noise)*torch.sqrt(dt)

def pf_ode_stepper(sde, score_fn):
    def v(xx, tt): return sde.pf_ode_vecfield(xx, tt, score_fn)
    def step(x, t, dt):
        k1 = v(x, t)
        k2 = v(x + dt*k1, t - dt)
        return x + 0.5*dt*(k1 + k2)
    return step
