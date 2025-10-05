import torch

def propose_and_accept_jumps(x, num_jumps, proposal_sampler):
    """
    Minimal jump adder for 1D signals:
    - proposal_sampler returns (B,1,L) 'z' to add as a jump impulse
    - currently unconditional accept (placeholder for ratio-based accept)
    """
    B = x.size(0)
    out = x.clone()
    for b in range(B):
        m = int(num_jumps[b].item())
        for _ in range(m):
            z = proposal_sampler(x[b:b+1])
            out[b:b+1] = out[b:b+1] + z
    return out
