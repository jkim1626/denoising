# sde_denoise

## Layout

- synthetic_data/
  - additive|multiplicative|jump|combined : .npz files with noisy/clean arrays
- sde_denoise/ : python package with SDEs, score nets, training & sampling
- scripts/ : entrypoints
- configs/ : minimal YAML configs

## Quickstart

pip install -r requirements.txt

# Train (additive VP example)

python scripts/train.py --config configs/additive_vp.yaml

# Sample with PF-ODE (no jumps)

python -m scripts.train --config configs/additive_vp.yaml

# Sample with reverse SDE (supports jumps)

python -m scripts.sample \
 --ckpt runs/additive/latest.pt \
 --mode pf_ode \
 --steps 40 \
 --shape 1 2048 \
 --noise_kind additive
