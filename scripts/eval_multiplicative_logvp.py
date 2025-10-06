import argparse, glob, os, statistics as st
import numpy as np
from subprocess import run

def snr_db(clean, x):
    num = float((clean**2).sum()) + 1e-12
    den = float(((x-clean)**2).sum()) + 1e-12
    return 10.0 * np.log10(num/den)

def psnr_db(clean, x):
    peak = float(np.max(np.abs(clean))) + 1e-12
    mse  = float(np.mean((x - clean)**2)) + 1e-12
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz in {args.folder}")

    out_dir = os.path.join(args.folder, "denoised_mult_logvp")
    os.makedirs(out_dir, exist_ok=True)

    for f in files:
        run(["python", "-m", "scripts.denoise_npz_multiplicative_logvp",
             "--ckpt", args.ckpt, "--npz", f], check=True)

    d_files = sorted(glob.glob(os.path.join(out_dir, "*_mult_logvp_denoised.npz")))
    base_map = {os.path.basename(p).replace("_mult_logvp_denoised",""): p for p in d_files}

    mse_n, mse_d, sin_list, sout_list, pin_list, pout_list = [], [], [], [], [], []
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
        sin_list.append(snr_db(clean, noisy))
        sout_list.append(snr_db(clean, den))
        pin_list.append(psnr_db(clean, noisy))
        pout_list.append(psnr_db(clean, den))

    print("Files evaluated:", len(mse_d))
    if mse_d:
        print(f"MSE (noisy):   mean={st.mean(mse_n):.4e}")
        print(f"MSE (denoised):mean={st.mean(mse_d):.4e}")
        print(f"SNR_in (dB):   mean={st.mean(sin_list):.2f}")
        print(f"SNR_out (dB):  mean={st.mean(sout_list):.2f}")
        print(f"ΔSNR (dB):     mean={st.mean([o-i for i,o in zip(sin_list,sout_list)]):.2f}")
        print(f"PSNR_in (dB):  mean={st.mean(pin_list):.2f}")
        print(f"PSNR_out (dB): mean={st.mean(pout_list):.2f}")
        print(f"ΔPSNR (dB):    mean={st.mean([o-i for i,o in zip(pin_list,pout_list)]):.2f}")

if __name__ == "__main__":
    main()