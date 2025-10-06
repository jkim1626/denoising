import argparse, glob, os, statistics as st, warnings
import numpy as np

# --- Optional deps
try:
    from bm3d import bm3d  
    _HAS_BM3D = True
except Exception:
    _HAS_BM3D = False

try:
    from scipy.signal import wiener as scipy_wiener  # pip install scipy
    _HAS_WIENER = True
except Exception:
    _HAS_WIENER = False

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

    # normalize
    scale = (nmax - nmin)
    z = (noisy - nmin) / scale

    # rough sigma estimate (normalized units) from first differences
    dif = np.diff(z)
    sigma_est = _mad_std(dif)

    # tile to 8xL
    H = 8
    z_img = np.tile(z[None, :], (H, 1))

    # bm3d call (some versions accept scalar sigma_psd)
    den_img = bm3d(z_img, sigma_psd=float(sigma_est))
    
    # collapse back to 1D by row mean
    den = den_img.mean(axis=0)

    # invert normalization
    return (den * scale + nmin).astype(np.float32)

def denoise_wiener_1d(noisy, mysize=7):
    """
    Wiener with noise power estimate. SciPy's Wiener can accept 'noise' kwarg
    (std^2). If not available in your version, it will be ignored.
    """
    if not _HAS_WIENER:
        raise RuntimeError("scipy not installed. `pip install scipy`")

    # estimate variance from high-frequency content
    dif = np.diff(noisy.astype(np.float64))
    sigma = _mad_std(dif)
    noise_var = float(sigma**2)

    try:
        out = scipy_wiener(noisy, mysize=mysize, noise=noise_var)
    except TypeError:
        # older SciPy without 'noise' arg
        out = scipy_wiener(noisy, mysize=mysize)

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True,
                    help="Folder with *.npz (expects 'clean' and 'noisy'). E.g. synthetic_data/additive")
    ap.add_argument("--methods", nargs="+", default=["bm3d", "wiener"],
                    choices=["bm3d", "wiener"])
    ap.add_argument("--wiener_window", type=int, default=7,
                    help="Odd window length for Wiener (default 7)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on #files to evaluate")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz in {args.folder}")
    if args.limit is not None:
        files = files[:args.limit]

    out_dirs = {m: _ensure_dir(os.path.join(args.folder, f"denoised_{m}")) for m in args.methods}

    metrics = {m: {"mse_den": [], "snr_out": [], "psnr": []} for m in args.methods}
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

        if "bm3d" in args.methods:
            if not _HAS_BM3D:
                warnings.warn("BM3D not available (pip install bm3d) — skipping.")
            else:
                try:
                    den = denoise_bm3d_1d(noisy)
                    np.savez(os.path.join(out_dirs["bm3d"], base + "_bm3d.npz"), denoised=den)
                    metrics["bm3d"]["mse_den"].append(float(np.mean((den - clean)**2)))
                    metrics["bm3d"]["snr_out"].append(snr_db(clean, den))
                    metrics["bm3d"]["psnr"].append(psnr_db(clean, den))
                except Exception as e:
                    warnings.warn(f"BM3D failed on {f}: {e}")

        if "wiener" in args.methods:
            if not _HAS_WIENER:
                warnings.warn("SciPy not available — skipping Wiener.")
            else:
                try:
                    den = denoise_wiener_1d(noisy, mysize=args.wiener_window)
                    np.savez(os.path.join(out_dirs["wiener"], base + "_wiener.npz"), denoised=den)
                    metrics["wiener"]["mse_den"].append(float(np.mean((den - clean)**2)))
                    metrics["wiener"]["snr_out"].append(snr_db(clean, den))
                    metrics["wiener"]["psnr"].append(psnr_db(clean, den))
                except Exception as e:
                    warnings.warn(f"Wiener failed on {f}: {e}")

    print(f"Files evaluated: {n_total}")
    if n_total:
        print(f"MSE(noisy): mean={st.mean(mse_noisy_list):.4e}")
        print(f"SNR_in  (dB): mean={st.mean(snr_in_list):.2f}")
        print(f"PSNR_in (dB): mean={st.mean(psnr_in_list):.2f}")

    for m in args.methods:
        arr = metrics[m]["mse_den"]
        if len(arr) == 0:
            print(f"\n[{m}] No results.")
            continue
        print(f"\n[{m.upper()}] over {len(arr)} files")
        print(f"MSE(den): mean={st.mean(metrics[m]['mse_den']):.4e}")
        print(f"SNR_out (dB): mean={st.mean(metrics[m]['snr_out']):.2f}")
        print(f"PSNR   (dB): mean={st.mean(metrics[m]['psnr']):.2f}")
        dsnr = st.mean([o - i for i, o in zip(snr_in_list[:len(metrics[m]['snr_out'])], metrics[m]['snr_out'])])
        print(f"ΔSNR   (dB): mean={dsnr:.2f}")

if __name__ == "__main__":
    main()