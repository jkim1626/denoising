# dataset_additive.py
import os, glob, numpy as np, torch
from torch.utils.data import Dataset

class AdditiveCleanDataset(Dataset):
    """
    Loads clean 1-D signals from synthetic_data/additive/train *.npz.
    We’ll add Gaussian noise *on the fly* during training.
    """
    def __init__(self, root="synthetic_data/additive/train"):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as z:
            x0 = z["clean"].astype(np.float32)   # shape [T]
        return torch.from_numpy(x0)              # tensor [T]
