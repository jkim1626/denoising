import os, glob, numpy as np, torch
from torch.utils.data import Dataset

class NPZFolder1D(Dataset):
    """
    Loads 1D spectra from synthetic_data/<noise_kind>/*.npz
    Expected keys:
      - 'clean': (L,) float (required)
      - 'noisy': (L,) float (optional, returned for eval)
      - 'fs': scalar (optional)
      - 'meta_json': string (optional)
    Returns:
      x0: (1, L) torch.float32  (channel-first)
      meta: dict with any available extra info (may be empty)
    """
    def __init__(self, root="synthetic_data", noise_kind="additive", dtype=torch.float32):
        assert noise_kind in ["additive", "multiplicative", "jump", "combined"]
        self.files = sorted(glob.glob(os.path.join(root, noise_kind, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz found in {root}/{noise_kind}")
        self.dtype = dtype

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path, allow_pickle=True)
        if "clean" not in d:
            raise KeyError(f"{path} missing 'clean' array.")
        x0 = d["clean"].astype(np.float32)    # (L,)
        x0 = torch.from_numpy(x0[None, :]).to(self.dtype)  # (1, L)
        meta = {}
        if "noisy" in d: meta["noisy"] = torch.from_numpy(d["noisy"].astype(np.float32))[None, :]
        if "fs" in d: meta["fs"] = float(np.array(d["fs"]))
        if "meta_json" in d: meta["meta_json"] = str(np.array(d["meta_json"]))
        return x0, meta
