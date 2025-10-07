import os, json
from pathlib import Path
import numpy as np

NOISE_TYPE       = "additive"                   
SNR_LIST_DB      = [0, 5, 10, 15, 20, 25]
FILES_PER_SNR    = 50
FS_HZ            = 1000.0
LENGTH_T         = 2048
SEED             = 9999

def rms(x): 
    return float(np.sqrt(np.mean(np.square(np.abs(x)))))

def make_time(T, fs): 
    return np.arange(T, dtype=np.float32) / float(fs)

def gen_clean_signal(T, fs, rng):
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

def add_additive(clean, snr_db, rng):
    sigma = rms(clean) / (10**(snr_db/20))
    noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    noisy = (clean + noise).astype(np.float32)
    return noisy, {"sigma": float(sigma)}

def save_npz(path, noisy, clean, fs, meta):
    meta_json = json.dumps(meta, separators=(",", ":"))
    np.savez_compressed(path, noisy=noisy.astype(np.float32), clean=clean.astype(np.float32),
                        fs=float(fs), meta_json=meta_json)

def run_generation(noise_type, snr_list, per_snr, outdir, fs, T, seed):
    rng = np.random.default_rng(seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    counter = 0
    for snr_db in snr_list:
        for _ in range(per_snr):
            clean = gen_clean_signal(T, fs, rng)
            noisy, params = add_additive(clean, snr_db, rng)

            realized = 20*np.log10( rms(clean) / max(1e-12, rms(noisy - clean)) )
            meta = {
                "domain": "generic",
                "noise_type": noise_type,
                "snr_db": float(snr_db),
                "realized_snr_db": float(realized),
                "source": "synthetic_validation",
                "seed": int(rng.integers(0, 2**31-1)),
                "gen_commit": "n/a",
                "params": params
            }
            fname = f"generic__{noise_type}__snr{int(round(snr_db))}__id{counter:05d}.npz"
            save_npz(Path(outdir)/fname, noisy, clean, fs, meta)
            counter += 1
    print(f"[OK] Wrote {counter} files to {Path(outdir).resolve()}")

if __name__ == "__main__":
    OUTDIR = "validation_additive"
    os.makedirs(OUTDIR, exist_ok=True)
    run_generation(NOISE_TYPE, SNR_LIST_DB, FILES_PER_SNR, OUTDIR, FS_HZ, LENGTH_T, SEED)