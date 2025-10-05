import torch

def heun_step(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + dt*k1, t - dt)
    return y + 0.5*dt*(k1 + k2)

def pf_ode_field(sde, score_fn):
    return lambda x, t: sde.pf_ode_vecfield(x, t, score_fn)
