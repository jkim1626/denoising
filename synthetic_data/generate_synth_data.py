# Simple synthetic 1-D denoising dataset generator (NPZ files)
# Edit CONFIG below 

import os, json
from pathlib import Path
import numpy as np

# ========== CONFIG (edit these) ==========
# "additive" | "multiplicative" | "jump" | "combined"
NOISE_TYPE       = "additive"                   
SNR_LIST_DB      = [0, 5, 10, 15, 20, 25]
FILES_PER_SNR    = 50
FS_HZ            = 1000.0                   # sampling rate
LENGTH_T         = 2048                     # number of samples
SEED             = 1234                     # global seed (reproducible)

# Noise knobs (safe defaults; tweak if desired)
SIGMA_M_GUESS    = 0.2                   # multiplicative (log-space) initial scale
IMPULSE_RATE     = 5e-4                  # expected impulses per sample (jump/combined)
IMPULSE_SCALE    = 1.0                   # amplitude scale for impulses
COMB_ADD_FRAC    = 0.7                   # additive fraction in "combined" (rest goes to multiplicative; small slice to impulses)
# ========================================

# ---------------- Utilities ----------------
def rms(x): 
    return float(np.sqrt(np.mean(np.square(np.abs(x)))))

def make_time(T, fs): 
    return np.arange(T, dtype=np.float32) / float(fs)

def gen_clean_signal(T, fs, rng):
    """Small, diverse bank of 1-D clean signals; normalized to RMS=1."""
    t = make_time(T, fs)
    pick = int(rng.integers(0, 8))
    if pick == 0:
        f  = float(rng.uniform(2, 50)); a = float(rng.uniform(0.6, 1.0)); ph = float(rng.uniform(0, 2*np.pi))
        x = a * np.sin(2*np.pi*f*t + ph)
    elif pick == 1:
        K = int(rng.integers(2, 5)); x = np.zeros_like(t); 
        for _ in range(K):
            f = float(rng.uniform(2, 200)); a = float(rng.uniform(0.2, 0.8)); ph = float(rng.uniform(0, 2*np.pi))
            x += a * np.sin(2*np.pi*f*t + ph)
        x /= max(1, K//2)
    elif pick == 2:
        f = float(rng.uniform(10, 200)); a = float(rng.uniform(0.8, 1.2)); decay = float(rng.uniform(0.5, 5.0))
        x = a * np.exp(-decay*t) * np.sin(2*np.pi*f*t)
    elif pick == 3:
        f0 = float(rng.uniform(2, 20)); f1 = float(rng.uniform(100, 400)); k = (f1 - f0) / (t[-1] if t[-1] > 0 else 1)
        phase = 2*np.pi*(f0*t + 0.5*k*np.square(t)); x = np.sin(phase)
    elif pick == 4:
        fc = float(rng.uniform(40, 120)); fm = float(rng.uniform(0.5, 3.0)); m = float(rng.uniform(0.2, 0.8))
        x = (1 + m*np.sin(2*np.pi*fm*t)) * np.sin(2*np.pi*fc*t)
    elif pick == 5:
        x = np.zeros_like(t); n_kinks = int(rng.integers(3, 6))
        idxs = np.sort(rng.integers(1, len(t)-1, size=n_kinks)); vals = rng.uniform(-1, 1, size=n_kinks+1); last = 0
        for i, idx in enumerate(np.append(idxs, len(t))):
            x[last:idx] = vals[i]; last = idx
        for _ in range(2):
            a = int(rng.integers(0, len(t)-100)); b = int(min(len(t), a + rng.integers(50, 300)))
            x[a:b] += np.linspace(0, float(rng.uniform(-0.5, 0.5)), b-a, dtype=np.float32)
    elif pick == 6:
        alpha = float(rng.uniform(0.85, 0.99)); e = rng.normal(0, 1, size=len(t)).astype(np.float32); x = np.zeros_like(t)
        for i in range(1, len(t)): x[i] = alpha*x[i-1] + e[i]
        x /= max(1e-6, np.std(x))
    else:
        x = np.sin(2*np.pi*float(rng.uniform(2, 20))*t); m = int(rng.integers(3, 12))
        idxs = rng.integers(0, len(t), size=m); amps = rng.uniform(-1.0, 1.0, size=m); x[idxs] += amps
    x = x.astype(np.float32); x /= max(1e-6, rms(x))
    return x

def calibrate_to_snr(clean, base_noise, snr_db):
    """Scale base_noise so that empirical SNR(clean, clean+scaled) matches target."""
    sig_r = rms(clean); target_noise_rms = sig_r / (10**(snr_db/20))
    base_r = rms(base_noise); scale = 0.0 if base_r < 1e-12 else target_noise_rms / base_r
    noisy = clean + scale * base_noise
    return noisy.astype(np.float32), {"scale": float(scale), "target_noise_rms": float(target_noise_rms)}

# --------------- Noise models ---------------
def add_additive(clean, snr_db, rng):
    sigma = rms(clean) / (10**(snr_db/20))
    noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    noisy = (clean + noise).astype(np.float32)
    return noisy, {"sigma": float(sigma)}

def add_multiplicative(clean, snr_db, rng, sigma_m_guess=0.2):
    eps = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32)
    base = clean * (np.exp(sigma_m_guess*eps) - 1.0).astype(np.float32)
    noisy, calib = calibrate_to_snr(clean, base, snr_db)
    calib.update({"sigma_m_guess": float(sigma_m_guess)})
    return noisy, calib

def add_jump(clean, snr_db, rng, impulse_rate=5e-4, amp_scale=1.0):
    T = clean.shape[0]; M = int(rng.poisson(impulse_rate*T))
    noise = np.zeros_like(clean, dtype=np.float32)
    if M > 0:
        idxs = rng.integers(0, T, size=M)
        noise[idxs] += rng.laplace(0.0, amp_scale, size=M).astype(np.float32)
    noisy, calib = calibrate_to_snr(clean, noise, snr_db)
    calib.update({"impulse_rate": float(impulse_rate), "impulse_count": int(M), "amp_scale": float(amp_scale)})
    return noisy, calib

def add_combined(clean, snr_db, rng, additive_frac=0.7, sigma_m_guess=0.2, impulse_rate=0.0, amp_scale=1.0):
    add_base  = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32)
    eps       = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32)
    mult_base = clean * (np.exp(sigma_m_guess*eps) - 1.0).astype(np.float32)
    T = clean.shape[0]; imp = np.zeros_like(clean, dtype=np.float32); M = 0
    if impulse_rate > 0.0:
        M = int(rng.poisson(impulse_rate*T))
        if M > 0:
            idxs = rng.integers(0, T, size=M)
            imp[idxs] += rng.laplace(0.0, amp_scale, size=M).astype(np.float32)
    def unit(v): r = rms(v); return v / (r + 1e-12)
    add_c, mult_c, imp_c = unit(add_base), unit(mult_base), (unit(imp) if M > 0 else imp)
    rem = 1.0 - additive_frac; mult_frac = rem if M == 0 else rem*0.85; imp_frac = 0.0 if M == 0 else rem*0.15
    base = additive_frac*add_c + mult_frac*mult_c + imp_frac*imp_c
    noisy, calib = calibrate_to_snr(clean, base, snr_db)
    calib.update({"additive_frac": float(additive_frac), "sigma_m_guess": float(sigma_m_guess),
                  "impulse_rate": float(impulse_rate), "impulse_count": int(M), "amp_scale": float(amp_scale)})
    return noisy, calib

# --------------- Save helper ---------------
def save_npz(path, noisy, clean, fs, meta):
    meta_json = json.dumps(meta, separators=(",", ":"))
    np.savez_compressed(path, noisy=noisy.astype(np.float32), clean=clean.astype(np.float32),
                        fs=float(fs), meta_json=meta_json)

# --------------- Main gen loop --------------
def run_generation(noise_type, snr_list, per_snr, outdir, fs, T, seed):
    rng = np.random.default_rng(seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    counter = 0
    for snr_db in snr_list:
        for _ in range(per_snr):
            clean = gen_clean_signal(T, fs, rng)
            if noise_type == "additive":
                noisy, params = add_additive(clean, snr_db, rng)
            elif noise_type == "multiplicative":
                noisy, params = add_multiplicative(clean, snr_db, rng, sigma_m_guess=SIGMA_M_GUESS)
            elif noise_type == "jump":
                noisy, params = add_jump(clean, snr_db, rng, impulse_rate=IMPULSE_RATE, amp_scale=IMPULSE_SCALE)
            elif noise_type == "combined":
                noisy, params = add_combined(clean, snr_db, rng,
                                             additive_frac=COMB_ADD_FRAC, sigma_m_guess=SIGMA_M_GUESS,
                                             impulse_rate=IMPULSE_RATE, amp_scale=IMPULSE_SCALE)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")

            realized = 20*np.log10( rms(clean) / max(1e-12, rms(noisy - clean)) )
            meta = {
                "domain": "generic",
                "noise_type": noise_type,
                "snr_db": float(snr_db),
                "realized_snr_db": float(realized),
                "source": "synthetic",
                "seed": int(rng.integers(0, 2**31-1)),
                "gen_commit": "n/a",
                "params": params
            }
            fname = f"generic__{noise_type}__snr{int(round(snr_db))}__id{counter:05d}.npz"
            save_npz(Path(outdir)/fname, noisy, clean, fs, meta)
            counter += 1
    print(f"[OK] Wrote {counter} files to {Path(outdir).resolve()}")

# --------------- Entry point ----------------
if __name__ == "__main__":

    # Make parent folder name match noise type by default
    OUTDIR = NOISE_TYPE
    os.makedirs(OUTDIR, exist_ok=True)
    run_generation(NOISE_TYPE, SNR_LIST_DB, FILES_PER_SNR, OUTDIR, FS_HZ, LENGTH_T, SEED)