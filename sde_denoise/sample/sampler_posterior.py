import torch

def pf_ode_stepper_posterior(sde, score_fn, like_grad_fn, like_weight=1.0, score_scale=1.0):
    """
    Heun step for posterior vector field:
      v_post(x,t) = [f - ∇·a - a * (score_scale * sθ)] + like_weight * ∇_x log p(y|x)
    """
    def v(xx, tt):
        prior_v = sde.pf_ode_vecfield(xx, tt, lambda x,t: score_scale * score_fn(x,t))
        lg = like_grad_fn(xx, tt)
        return prior_v + like_weight * lg

    def step(x, t, dt):
        k1 = v(x, t)
        k2 = v(x + dt*k1, t - dt)
        return x + 0.5*dt*(k1 + k2)
    return step
