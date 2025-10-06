import argparse, glob, os, statistics as st, subprocess, sys, numpy as np

def snr_db(clean, x):
    num = float((clean**2).sum()) + 1e-12
    den = float(((x-clean)**2).sum()) + 1e-12
    return 10.0 * np.log10(num/den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--folder", required=True)  # synthetic_data/additive
    ap.add_argument("--scales", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz in {args.folder}")

    # resolve how to call the single-file denoiser
    module_name = "scripts.denoise_npz_calibrated"
    script_path = os.path.join(os.path.dirname(__file__), "denoise_npz_calibrated.py")
    use_module = os.path.exists(script_path)  # module will work if script exists under scripts/

    best = None
    for c in args.scales:
        # batch denoise
        for f in files:
            if use_module:
                cmd = [sys.executable, "-m", module_name, "--ckpt", args.ckpt, "--npz", f, "--score_scale", str(c)]
            else:
                cmd = [sys.executable, script_path, "--ckpt", args.ckpt, "--npz", f, "--score_scale", str(c)]
            subprocess.run(cmd, check=True)

        # gather outputs
        out_dir = os.path.join(args.folder, "denoised_cal")
        d_files = sorted(glob.glob(os.path.join(out_dir, f"*_cal{c:.2f}.npz")))
        base_map = {os.path.basename(p).split("_cal")[0]+".npz": p for p in d_files}

        mse_n, mse_d, sin, sout = [], [], [], []
        for f in files:
            base = os.path.basename(f)
            if base not in base_map: continue
            d0 = np.load(f, allow_pickle=True)
            if "clean" not in d0 or "noisy" not in d0: continue
            den = np.load(base_map[base])["denoised"].astype(np.float32)
            clean = d0["clean"].astype(np.float32)
            noisy = d0["noisy"].astype(np.float32)
            mse_n.append(float(np.mean((noisy-clean)**2)))
            mse_d.append(float(np.mean((den-clean)**2)))
            sin.append(snr_db(clean, noisy))
            sout.append(snr_db(clean, den))

        if mse_d:
            dsnr = st.mean([o - i for i, o in zip(sin, sout)])
            print(f"[scale {c:.2f}] Files={len(mse_d)}  ΔSNR={dsnr:.2f} dB  "
                  f"MSE(noisy)={st.mean(mse_n):.4e}  MSE(den)={st.mean(mse_d):.4e}")
            if best is None or dsnr > best[0]:
                best = (dsnr, c)

    if best:
        print(f"BEST ΔSNR={best[0]:.2f} dB @ score_scale={best[1]:.2f}")

if __name__ == "__main__":
    main()
