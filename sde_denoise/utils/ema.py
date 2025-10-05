import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.requires_grad}
        self.model = model

    def update(self, model=None):
        if model is None:
            model = self.model
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model=None):
        if model is None:
            model = self.model
        for k, v in model.state_dict().items():
            if k in self.shadow:
                v.data.copy_(self.shadow[k].data)
