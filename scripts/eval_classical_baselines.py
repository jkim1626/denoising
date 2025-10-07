import argparse, glob, os, statistics as st, warnings
import numpy as np

try:
    from bm3d import bm3d  
    _HAS_BM3D = True
except Exception:
    _HAS_BM3D = False

def snr_db(clean, x):
    num = float((clean**2).sum()) + 1e-12
    den = float(((x - clean)**2).sum()) + 1e-12
    return 10.0 * np.log10(num / den)

def psnr_db(clean, x):
    peak = float(np.max(np.abs(clean))) + 1e-12
    mse  = float(np.mean((x - clean)**2)) + 1e-12
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _mad_std(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * (mad + 1e-12)

def denoise_bm3d_1d(noisy):
    """
    Adapt 1D -> 2D by tiling to 8 rows (min block size), run BM3D, average rows back.
    Normalize to [0,1] per-signal (BM3D convention), then invert.
    """
    if not _HAS_BM3D:
        raise RuntimeError("bm3d package not installed. `pip install bm3d`")

    nmin, nmax = float(np.min(noisy)), float(np.max(noisy))
    if nmax <= nmin + 1e-12:
        return noisy.copy()

    scale = (nmax - nmin)
    z = (noisy - nmin) / scale

    dif = np.diff(z)
    sigma_est = _mad_std(dif)

    H = 8
    z_img = np.tile(z[None, :], (H, 1))

    den_img = bm3d(z_img, sigma_psd=float(sigma_est))
    
    den = den_img.mean(axis=0)

    return (den * scale + nmin).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True,
                    help="Folder with *.npz (expects 'clean' and 'noisy')")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on #files to evaluate")
    args = ap.parse_args()

    if not _HAS_BM3D:
        raise RuntimeError("BM3D not available. Install with: pip install bm3d")

    files = sorted(glob.glob(os.path.join(args.folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz in {args.folder}")
    if args.limit is not None:
        files = files[:args.limit]

    out_dir = _ensure_dir(os.path.join(args.folder, "denoised_bm3d"))

    mse_den, snr_out, psnr_out = [], [], []
    mse_noisy_list, snr_in_list, psnr_in_list = [], [], []

    n_total = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "clean" not in d or "noisy" not in d:
            warnings.warn(f"Skipping (missing 'clean'/'noisy'): {f}")
            continue

        clean = d["clean"].astype(np.float32)
        noisy = d["noisy"].astype(np.float32)

        if clean.ndim != 1 or noisy.ndim != 1 or clean.shape != noisy.shape:
            warnings.warn(f"Expected 1D same-shape arrays in {f}; got clean{clean.shape}, noisy{noisy.shape}")
            continue

        mse_noisy_list.append(float(np.mean((noisy - clean)**2)))
        snr_in_list.append(snr_db(clean, noisy))
        psnr_in_list.append(psnr_db(clean, noisy))

        base = os.path.basename(f).replace(".npz", "")
        n_total += 1

        try:
            den = denoise_bm3d_1d(noisy)
            np.savez(os.path.join(out_dir, base + "_bm3d.npz"), denoised=den)
            mse_den.append(float(np.mean((den - clean)**2)))
            snr_out.append(snr_db(clean, den))
            psnr_out.append(psnr_db(clean, den))
        except Exception as e:
            warnings.warn(f"BM3D failed on {f}: {e}")

    print(f"Files evaluated: {n_total}")
    if n_total:
        print(f"MSE(noisy): mean={st.mean(mse_noisy_list):.4e}")
        print(f"SNR_in  (dB): mean={st.mean(snr_in_list):.2f}")
        print(f"PSNR_in (dB): mean={st.mean(psnr_in_list):.2f}")

    if len(mse_den) == 0:
        print(f"\n[BM3D] No results.")
    else:
        print(f"\n[BM3D] over {len(mse_den)} files")
        print(f"MSE(den): mean={st.mean(mse_den):.4e}")
        print(f"SNR_out (dB): mean={st.mean(snr_out):.2f}")
        print(f"PSNR   (dB): mean={st.mean(psnr_out):.2f}")
        dsnr = st.mean([o - i for i, o in zip(snr_in_list[:len(snr_out)], snr_out)])
        dpsnr = st.mean([o - i for i, o in zip(psnr_in_list[:len(psnr_out)], psnr_out)])
        print(f"ΔSNR   (dB): mean={dsnr:.2f}")
        print(f"ΔPSNR  (dB): mean={dpsnr:.2f}")

if __name__ == "__main__":
    main()