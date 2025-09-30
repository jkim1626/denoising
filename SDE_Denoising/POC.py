# POC script 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# --- CONFIGURATION AND STYLING ---
D = 200      # Dimension of the signal
T = 1.0      # Total time
N = 1000     # Number of time steps
beta_min = 0.1
beta_max = 35.0 

style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (18, 12) # Adjusted for 2x2 layout

# --- SHARED FUNCTIONS AND NOISE SCHEDULE ---
dt = T / N
ts = np.linspace(0, T, N + 1)

# Pre-compute noise schedule (Variance Preserving)
betas_sde = beta_min + 0.5 * (beta_max - beta_min) * (1 - np.cos(np.pi * ts / T))
alphas_sde = np.exp(-0.5 * np.cumsum(betas_sde) * dt)
sigmas_sde = np.sqrt(1 - alphas_sde**2)

# Generate the clean signal
x_axis = np.linspace(0, 10, D)
x0 = np.sin(x_axis) * np.cos(x_axis * 0.5)
x0[int(D * 0.2):int(D * 0.25)] += 0.8
x0[int(D * 0.7):int(D * 0.72)] -= 1.0

def plot_results(ax, title, noisy_x, denoised_x, color):
    """Helper function to plot the results on a given axis."""
    ax.plot(x0, color='blue', linewidth=2, label='Clean Signal', alpha=0.9, zorder=2)
    ax.plot(noisy_x, color='gray', linestyle='-', linewidth=1.5, label='Noisy Signal', alpha=0.7, zorder=1)
    ax.plot(denoised_x, color=color, linewidth=3, linestyle='--', label='Denoised Signal', zorder=3)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel("Signal Dimension")
    ax.set_ylabel("Amplitude")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- 1. ADDITIVE GAUSSIAN NOISE ---

def get_noisy_additive(t_idx, base_signal=None):
    """Adds additive Gaussian noise for a given time step index."""
    if base_signal is None:
        base_signal = x0
    alpha_t = alphas_sde[t_idx]
    sigma_t = sigmas_sde[t_idx]
    noise = np.random.randn(D) * sigma_t
    return alpha_t * base_signal + noise

def oracle_score_additive(xt, t_idx, target_signal=None):
    """The perfect 'oracle' score function for additive noise."""
    if target_signal is None:
        target_signal = x0
    alpha_t = alphas_sde[t_idx]
    sigma_t = sigmas_sde[t_idx]
    if sigma_t < 1e-5:
        return np.zeros_like(xt)
    return -(xt - alpha_t * target_signal) / (sigma_t**2)

def denoise_additive(noisy_x, target_signal=None):
    """
    Runs the reverse process using a stable ancestral sampler (DDPM-style).
    """
    if target_signal is None:
        target_signal = x0
        
    xt = np.copy(noisy_x)
    alpha_bars = alphas_sde**2
    
    for i in range(N, 0, -1):
        z = np.random.randn(D) if i > 1 else np.zeros(D)
        
        score = oracle_score_additive(xt, i, target_signal=target_signal)
        epsilon_pred = -sigmas_sde[i] * score
        x0_pred = (xt - sigmas_sde[i] * epsilon_pred) / alphas_sde[i]
        x0_pred = np.clip(x0_pred, -2.5, 2.5)

        alpha_bar_prev = alpha_bars[i-1]
        
        coeff1 = np.sqrt(alpha_bar_prev) * (1 - alpha_bars[i] / alpha_bar_prev) / (1 - alpha_bars[i])
        coeff2 = np.sqrt(alphas_sde[i]**2 / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bars[i])
        posterior_mean = coeff1 * x0_pred + coeff2 * xt

        posterior_variance = (1 - alpha_bars[i] / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bars[i])

        xt = posterior_mean + np.sqrt(np.maximum(posterior_variance, 1e-9)) * z
        
    return xt

# --- 2. MULTIPLICATIVE (SPECKLE) NOISE ---

def get_noisy_multiplicative(t_idx):
    log_x0 = np.log(np.abs(x0) + 1e-6)
    Lambda_t = np.sum(betas_sde[:t_idx+1]) * dt
    noise = np.random.randn(D) * np.sqrt(Lambda_t)
    log_xt = log_x0 - 0.5 * Lambda_t + noise
    return np.exp(log_xt) * np.sign(x0)

def denoise_multiplicative(noisy_x):
    """
    FIXED: Now correctly uses the sign of the ground-truth signal x0,
    instead of the sign of the noisy input. This is critical for pipelines.
    """
    # Use the known, true sign from the original signal.
    signs = np.sign(x0)
    
    yt = np.log(np.abs(noisy_x) + 1e-6)
    log_x0 = np.log(np.abs(x0) + 1e-6)

    for i in range(N, 0, -1):
        z = np.random.randn(D) if i > 1 else np.zeros(D)
        
        Lambda_i = np.sum(betas_sde[:i+1]) * dt
        Lambda_i_minus_1 = np.sum(betas_sde[:i]) * dt
        var_i, var_i_minus_1 = Lambda_i, Lambda_i_minus_1
        
        mean_pred = (var_i_minus_1 * yt + (var_i - var_i_minus_1) * log_x0) / (var_i + 1e-9)
        variance = (var_i - var_i_minus_1) * var_i_minus_1 / (var_i + 1e-9)
        
        yt = mean_pred + np.sqrt(np.maximum(variance, 1e-9)) * z

    denoised_magnitude = np.exp(yt)
    return denoised_magnitude * signs

# --- 3. IMPULSIVE (JUMP) NOISE ---

jump_locations = []

def get_noisy_jump(t_idx):
    global jump_locations
    sigma_t = sigmas_sde[t_idx] * 0.5
    noise = np.random.randn(D) * sigma_t
    xt = alphas_sde[t_idx] * x0 + noise
    
    jump_locations = []
    num_jumps = 7
    for _ in range(num_jumps):
        loc = np.random.randint(0, D)
        mag = (np.random.choice([-1, 1])) * (1.5 + np.random.rand() * 2)
        xt[loc] += mag
        jump_locations.append(loc)
    return xt

def denoise_jump(noisy_x):
    xt = np.copy(noisy_x)
    alpha_bars = alphas_sde**2
    
    for i in range(N, 0, -1):
        z = np.random.randn(D) if i > 1 else np.zeros(D)
        
        score = oracle_score_additive(xt, i)
        epsilon_pred = -sigmas_sde[i] * score
        x0_pred = (xt - sigmas_sde[i] * epsilon_pred) / alphas_sde[i]

        for loc in jump_locations:
             x0_pred[loc] = x0[loc]

        x0_pred = np.clip(x0_pred, -2.5, 2.5)

        alpha_bar_prev = alpha_bars[i-1]
        
        coeff1 = np.sqrt(alpha_bar_prev) * (1 - alpha_bars[i] / alpha_bar_prev) / (1 - alpha_bars[i])
        coeff2 = np.sqrt(alphas_sde[i]**2 / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bars[i])
        posterior_mean = coeff1 * x0_pred + coeff2 * xt

        posterior_variance = (1 - alpha_bars[i] / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bars[i])

        xt = posterior_mean + np.sqrt(np.maximum(posterior_variance, 1e-9)) * z
        
    return xt

# --- 4. NEW: COMBINED ADDITIVE + MULTIPLICATIVE NOISE ---
def get_noisy_combined(t_idx):
    """Applies multiplicative noise, then additive noise on top."""
    # Stage 1: Create signal with multiplicative noise
    x_mult = get_noisy_multiplicative(t_idx)
    # Stage 2: Use the result as a base for additive noise
    x_combined = get_noisy_additive(t_idx, base_signal=x_mult)
    return x_combined, x_mult # Return intermediate for oracle

def denoise_combined(noisy_x, intermediate_signal):
    """Denoise in a two-stage pipeline."""
    print("  Stage 1: Denoising additive component...")
    # Step 1: Remove additive noise, targeting the intermediate speckled signal
    denoised_additive_stage = denoise_additive(noisy_x, target_signal=intermediate_signal)
    
    print("  Stage 2: Denoising multiplicative component...")
    # Step 2: Remove multiplicative noise from the result of stage 1
    final_denoised = denoise_multiplicative(denoised_additive_stage)
    
    return final_denoised

# --- MAIN EXECUTION AND PLOTTING ---
if __name__ == '__main__':
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Score-Based SDE Denoising Demonstrations', fontsize=22, fontweight='bold')

    print("1. Running Additive Denoising...")
    noisy_additive = get_noisy_additive(N)
    denoised_additive = denoise_additive(noisy_additive)
    plot_results(ax1, 'Additive Gaussian Noise', noisy_additive, denoised_additive, 'green')

    print("\n2. Running Multiplicative (Speckle) Denoising...")
    noisy_multiplicative = get_noisy_multiplicative(N)
    denoised_multiplicative = denoise_multiplicative(noisy_multiplicative)
    plot_results(ax2, 'Multiplicative (Speckle) Noise', noisy_multiplicative, denoised_multiplicative, 'red')

    print("\n3. Running Impulsive (Jump) Denoising...")
    noisy_jump = get_noisy_jump(N)
    denoised_jump = denoise_jump(noisy_jump)
    plot_results(ax3, 'Impulsive (Jump) Noise', noisy_jump, denoised_jump, 'orange')
    
    print("\n4. Running Combined Noise Denoising...")
    noisy_combined, intermediate_mult = get_noisy_combined(N)
    denoised_combined = denoise_combined(noisy_combined, intermediate_mult)
    plot_results(ax4, 'Combined Additive + Multiplicative', noisy_combined, denoised_combined, 'purple')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


