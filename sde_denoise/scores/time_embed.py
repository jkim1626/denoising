import torch, torch.nn as nn, math

class LogSNREmbed(nn.Module):
    def __init__(self, dim=128, J=16):
        super().__init__()
        self.lin1 = nn.Linear(2*J, dim)
        self.act  = nn.SiLU()
        self.lin2 = nn.Linear(dim, dim)
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(10000.0), J)))

    def forward(self, lambda_t):
        lamb = lambda_t[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.cos(lamb), torch.sin(lamb)], dim=-1)
        h = self.lin2(self.act(self.lin1(emb)))
        return h
