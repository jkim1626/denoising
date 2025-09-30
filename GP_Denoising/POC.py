import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, Matern
from sklearn.preprocessing import StandardScaler
from scipy import signal

# Helper function for user input
def get_input(prompt, default, type_func=float):
    value = input(prompt + f" [default: {default}]: ").strip()
    return type_func(value) if value else default

# User inputs
k = int(get_input("Enter number of transients (k, e.g., 10)", 10, int))
f = get_input("Enter base signal frequency (Hz, e.g., 1)", 1.0)
sigma = get_input("Enter noise level (std dev or scale, e.g., 0.5)", 0.5)
fc = get_input("Enter low-pass filter cutoff frequency (Hz, e.g., 2)", 2.0)
noise_types = ["Wiener", "Poisson", "Levy", "Gaussian", "Pink", "OU Increments"]
print("Noise types:", ", ".join(f"{i+1}: {n}" for i, n in enumerate(noise_types)))
noise_choice = int(get_input("Enter noise type (1-6)", 1, int)) - 1
noise_type = noise_types[noise_choice]
signal_types = [
    "Single Sinusoid", "Multi-Freq Sinusoid", "Chirp", "Exp Decays", "Square Wave",
    "OFDM-like", "Gaussian Pulse", "NMR Spectrum", "Polynomial", "Damped Oscillation"
]
print("Signal types:", ", ".join(f"{i+1}: {s}" for i, s in enumerate(signal_types)))
signal_choice = int(get_input("Enter signal type (1-10)", 1, int)) - 1
signal_type = signal_types[signal_choice]
amp = get_input("Enter signal amplitude (e.g., 1)", 1.0)  # Fixed default, independent of sigma
duration = get_input("Enter time duration (seconds, e.g., 1)", 1.0)
use_periodic = int(get_input("Use periodic kernel? (1=yes, 0=no)", 1, int))

# Time points
fs = 1000
dt = 1 / fs
T = int(fs * duration)
t = np.linspace(dt, duration, T)

# Generate signal
if signal_type == "Single Sinusoid":
    s = amp * np.sin(2 * np.pi * f * t)
elif signal_type == "Multi-Freq Sinusoid":
    s = amp * (0.6 * np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t))
elif signal_type == "Chirp":
    k = 5 * f / duration
    s = amp * np.sin(2 * np.pi * (f + k * t) * t)
elif signal_type == "Exp Decays":
    s = amp * (0.5 * np.exp(-2 * t) * np.cos(2 * np.pi * f * t) + 
               0.5 * np.exp(-4 * t) * np.cos(2 * np.pi * 2 * f * t))
elif signal_type == "Square Wave":
    s = amp * np.sign(np.sin(2 * np.pi * f * t))
elif signal_type == "OFDM-like":
    s = amp * sum(0.3 * np.cos(2 * np.pi * (f + i * 1) * t + np.random.uniform(0, 2 * np.pi)) 
                  for i in range(5))
elif signal_type == "Gaussian Pulse":
    mu, sig = duration / 2, duration / 10
    s = amp * np.exp(-((t - mu) ** 2) / (2 * sig ** 2))
elif signal_type == "NMR Spectrum":
    # NMR spectrum as sum of 15 Lorentzian peaks in time domain
    width = duration  # Full time width (e.g., 1 s)
    linewidth = width / 20  # Linewidth ~1/20 of time width (e.g., 0.05 s)
    gamma = linewidth / 2  # Half-width at half-maximum (e.g., 0.025 s)
    num_peaks = 15
    peak_positions = np.random.uniform(0.1, 0.9, num_peaks)  # Random positions in [0.1, 0.9] s
    peak_amplitudes = np.random.uniform(0.5, 1.5, num_peaks)  # Random amplitudes between 0.5 and 1.5
    s = np.zeros(T)
    for pos, peak_amp in zip(peak_positions, peak_amplitudes):
        # Lorentzian: A * gamma / ((t - t0)^2 + gamma^2)
        s += peak_amp * (gamma / ((t - pos) ** 2 + gamma ** 2))
    # Rescale to ensure max peak height is 1.0
    s = s / np.max(s)  # Normalize so max height is 1.0
elif signal_type == "Polynomial":
    s = amp * (0.5 * t ** 2 - t + 0.2)
elif signal_type == "Damped Oscillation":
    s = amp * np.exp(-2 * t) * np.sin(2 * np.pi * f * t)

# Generate noise
noise = np.zeros((k, T), dtype=float)
if noise_type == "Wiener":
    noise = np.random.normal(0, sigma * np.sqrt(dt), size=(k, T))
elif noise_type == "Poisson":
    lam = sigma * dt
    noise = np.random.poisson(lam, size=(k, T)) - lam
elif noise_type == "Levy":
    noise = np.random.standard_cauchy(size=(k, T)) * sigma * dt
elif noise_type == "Gaussian":
    noise = np.random.normal(0, sigma, size=(k, T))
elif noise_type == "Pink":
    for i in range(k):
        white = np.random.normal(0, sigma, T)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(T, dt)
        fft[1:] /= np.sqrt(freqs[1:])
        noise[i] = np.fft.irfft(fft, T)
elif noise_type == "OU Increments":
    theta = 1.0
    dW = np.random.normal(0, sigma * np.sqrt(dt), size=(k, T))
    X_prev = np.zeros(k)
    for j in range(T):
        noise[:, j] = -theta * X_prev * dt + dW[:, j]
        X_prev += noise[:, j]

x = s + noise
x_bar = np.mean(x, axis=0)

# Robust preprocessing for specific noise types
if noise_type in ["Levy", "Pink"]:
    x_bar = np.median(x, axis=0)

# Normalize data
scaler = StandardScaler()
x_bar_scaled = scaler.fit_transform(x_bar.reshape(-1, 1)).flatten()

# GP Denoising with adaptive kernel
length_scale = 1 / (f if f > 0 else 1)
if use_periodic and signal_type == "Damped Oscillation":
    kernel = (ExpSineSquared(length_scale=length_scale, periodicity=1/f, length_scale_bounds=(1e-2, 1e2)) * 
              RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) + 
              WhiteKernel(noise_level=sigma / np.sqrt(k), noise_level_bounds=(1e-5, 1e1)))
elif use_periodic and signal_type in ["Single Sinusoid", "Multi-Freq Sinusoid"]:
    kernel = (ExpSineSquared(length_scale=length_scale, periodicity=1/f, length_scale_bounds=(1e-2, 1e2)) + 
              WhiteKernel(noise_level=sigma / np.sqrt(k), noise_level_bounds=(1e-5, 1e1)))
elif signal_type in ["Square Wave", "NMR Spectrum"]:
    kernel = (Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2), nu=1.5) + 
              WhiteKernel(noise_level=sigma / np.sqrt(k), noise_level_bounds=(1e-5, 1e1)))
else:
    kernel = (RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) + 
              WhiteKernel(noise_level=sigma / np.sqrt(k), noise_level_bounds=(1e-5, 1e1)))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
gp.fit(t.reshape(-1, 1), x_bar_scaled)
s_GP_scaled, _ = gp.predict(t.reshape(-1, 1), return_std=True)
s_GP = scaler.inverse_transform(s_GP_scaled.reshape(-1, 1)).flatten()

# Low-Pass Filtering
nyquist = fs / 2
Wn = fc / nyquist
if not (0 < Wn < 1):
    Wn = min(max(Wn, 0.01), 0.99)
b, a = signal.butter(4, Wn, btype='low')
s_LP = signal.filtfilt(b, a, x_bar)

# Metrics
mse_GP = np.mean((s_GP - s) ** 2)
mse_LP = np.mean((s_LP - s) ** 2)
snr_GP = 10 * np.log10(np.sum(s ** 2) / np.sum((s_GP - s) ** 2))
snr_LP = 10 * np.log10(np.sum(s ** 2) / np.sum((s_LP - s) ** 2))

print(f"\nSignal Type: {signal_type}")
print(f"Noise Type: {noise_type}")
print(f"GP MSE: {mse_GP:.4f}, SNR: {snr_GP:.2f} dB")
print(f"Low-Pass MSE: {mse_LP:.4f}, SNR: {snr_LP:.2f} dB")

# Plots
plt.figure(figsize=(15, 12))
plt.subplot(321)
plt.plot(t, s, 'b-', label='True Signal')
plt.plot(t, x_bar, 'r--', label='Noisy Average')
plt.title('True Signal vs Noisy Average')
plt.legend()

plt.subplot(322)
for i in range(min(k, 5)):
    plt.plot(t, x[i], 'g-', alpha=0.3, label='Transient' if i == 0 else "")
plt.title('Sample Transients')
plt.legend()

plt.subplot(323)
plt.plot(t, s, 'b-', label='True Signal')
plt.plot(t, s_GP, 'r--', label='GP Denoised')
plt.title('GP Denoised vs True Signal')
plt.legend()

plt.subplot(324)
plt.plot(t, s, 'b-', label='True Signal')
plt.plot(t, s_LP, 'r--', label='Low-Pass Filtered')
plt.title('Low-Pass Filtered vs True Signal')
plt.legend()

plt.subplot(325)
plt.plot(t, s_GP - s, 'r-', label='GP Residual')
plt.title('GP Residual')
plt.legend()

plt.subplot(326)
plt.plot(t, s_LP - s, 'r-', label='Low-Pass Residual')
plt.title('Low-Pass Residual')
plt.legend()

plt.tight_layout()
plt.show()

