from abc import ABC, abstractmethod
import torch

def matvec(a, v):
    return torch.matmul(a, v.unsqueeze(-1)).squeeze(-1)

class SDE(ABC):
    def __init__(self, T: float = 1.0, device: str = "cuda"):
        self.T = T
        self.device = device

    @abstractmethod
    def drift(self, x, t):
        ...

    @abstractmethod
    def diffusion_linop(self, x, t):
        """Return a function ξ -> Σ(x,t) ξ (linear map)."""
        ...

    @abstractmethod
    def a_matrix(self, x, t):
        """a = 1/2 Σ Σ^T, shape (..., d, d)."""
        ...

    def div_a(self, x, t):
        """∇·a (vector); default 0 for x-independent Σ."""
        return torch.zeros_like(x)

    def reverse_drift(self, x, t, score_fn):
        a = self.a_matrix(x, t)
        div_a = self.div_a(x, t)
        s = score_fn(x, t)
        return self.drift(x, t) - div_a - 2.0 * matvec(a, s)

    def pf_ode_vecfield(self, x, t, score_fn):
        a = self.a_matrix(x, t)
        div_a = self.div_a(x, t)
        s = score_fn(x, t)
        return self.drift(x, t) - div_a - matvec(a, s)
