import torch, os

def save_ckpt(model, ema, opt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "ema": ema.shadow,
                "opt": opt.state_dict()}, path)

def load_ckpt(model, ema, opt, path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"])
    ema.shadow = obj.get("ema", ema.shadow)
    if opt is not None and "opt" in obj:
        opt.load_state_dict(obj["opt"])
