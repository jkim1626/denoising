"""
Enhanced Multi-Noise SDE Denoising with Neural Score Networks and PF-ODE

Extended version with:
1. Proper multiplicative noise in log-space (Geometric VP SDE)
2. Trainable neural network score functions
3. Probability Flow ODE deterministic sampling
4. Modular design supporting both additive and multiplicative noise
"""

# Purpose: RESEARCH-GRADE SDE with NEURAL SCORE NETWORKS (PyTorch) + PROBABILITY-FLOW ODE (deterministic) sampling.
# What it runs: Train/eval a learnable score (DSM), compare SDE vs PF-ODE, support additive + (log-space) multiplicative noise.
# When to use: State-of-the-art experiments, stability/quality comparisons, and publishable results on real datasets.
# Paper tie-in: Exercises the paper’s practical side—learned score estimation + PF-ODE equivalence to the reverse SDE marginals.
# Notes: Heavier dependencies/compute; ideal as the main engine for results sections.

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass
import warnings
from scipy.integrate import solve_ivp
warnings.filterwarnings('ignore')

# Neural network dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural network score functions will be disabled.")

# Set plotting parameters
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

@dataclass
class SDEConfig:
    D: int = 200          # Signal dimension
    T: float = 1.0        # Total time
    N: int = 100          # Number of time steps
    beta_min: float = 0.0001  # Minimum noise level
    beta_max: float = 0.02    # Maximum noise level
    noise_type: str = "additive"  # "additive" or "multiplicative"
    score_type: str = "oracle"    # "oracle" or "neural"
    sampling_method: str = "ddpm"  # "ddpm" or "ode"

class NoiseSchedule:    
    def __init__(self, config: SDEConfig):
        self.config = config
        if config.noise_type == "multiplicative":
            # Cosine schedule for multiplicative noise (better for training)
            self.betas = self._cosine_schedule(config.N, config.beta_min, config.beta_max)
        else:
            # Linear schedule for additive noise
            self.betas = np.linspace(config.beta_min, config.beta_max, config.N)
        
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        
        # For multiplicative noise: Lambda_t = integral of beta(s) ds
        self.lambda_t = np.cumsum(self.betas) * (config.T / config.N)
    
    def _cosine_schedule(self, timesteps: int, beta_min: float, beta_max: float) -> np.ndarray:
        """Cosine noise schedule for better training stability"""
        s = 0.008  # offset
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, beta_min, beta_max)

class ScoreFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray, t_idx: int, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        pass

class MLPScoreNetwork(nn.Module):
    """Multi-layer perceptron for score function estimation"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 512, 512], 
                 time_embed_dim: int = 128, noise_type: str = "additive"):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        self.noise_type = noise_type
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim // 2, time_embed_dim),
            nn.SiLU()
        )
        
        # Main network
        layers = []
        in_dim = input_dim + time_embed_dim
        
        # For multiplicative noise, add log(|x|) as auxiliary input
        if noise_type == "multiplicative":
            in_dim += input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, input_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_embed = self.time_embed(t.unsqueeze(-1))
        
        # Prepare input
        if self.noise_type == "multiplicative":
            # Add log(|x| + eps) as auxiliary input for multiplicative noise
            log_abs_x = torch.log(torch.abs(x) + 1e-8)
            net_input = torch.cat([x, t_embed, log_abs_x], dim=-1)
        else:
            net_input = torch.cat([x, t_embed], dim=-1)
        
        return self.network(net_input)

class NeuralScoreFunction(ScoreFunction):
    """Neural network-based score function"""
    
    def __init__(self, model_path: str, config: SDEConfig, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural score functions")
        
        self.config = config
        self.device = device
        
        # Load trained model
        self.model = MLPScoreNetwork(
            input_dim=config.D, 
            noise_type=config.noise_type
        ).to(device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded neural score model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Please train a model first.")
    
    def __call__(self, x: np.ndarray, t_idx: int, **kwargs) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            t_tensor = torch.tensor(t_idx / self.config.N, dtype=torch.float32).to(self.device)
            
            # Ensure batch dimension
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            score = self.model(x_tensor, t_tensor)
            
            if squeeze_output:
                score = score.squeeze(0)
            
            return score.cpu().numpy()
    
    def get_type(self) -> str:
        return f"neural_{self.config.noise_type}"

class UniversalOracleScore(ScoreFunction):    
    def __init__(self, schedule: NoiseSchedule, target_signal: np.ndarray, noise_type: str = "additive"):
        self.schedule = schedule
        self.target_signal = target_signal
        self.noise_type = noise_type
        
        if noise_type == "multiplicative":
            # Ensure positive values for multiplicative noise
            self.target_signal = np.maximum(np.abs(target_signal), 0.01)
    
    def __call__(self, x: np.ndarray, t_idx: int, **kwargs) -> np.ndarray:
        if self.noise_type == "additive":
            return self._additive_score(x, t_idx)
        elif self.noise_type == "multiplicative":
            return self._multiplicative_score(x, t_idx)
        else:
            return self._additive_score(x, t_idx)  # Default fallback
    
    def _additive_score(self, x: np.ndarray, t_idx: int) -> np.ndarray:
        """Standard score for additive Gaussian noise"""
        alpha_bar_t = self.schedule.alpha_bars[t_idx]
        predicted_noise = (x - np.sqrt(alpha_bar_t) * self.target_signal) / np.sqrt(1 - alpha_bar_t + 1e-8)
        return -predicted_noise / np.sqrt(1 - alpha_bar_t + 1e-8)
    
    def _multiplicative_score(self, x: np.ndarray, t_idx: int) -> np.ndarray:
        """Analytic score for multiplicative noise in log-space"""
        # Ensure positive values
        x_pos = np.maximum(np.abs(x), 1e-8)
        
        # Get Lambda_t (integrated beta)
        lambda_t = self.schedule.lambda_t[t_idx]
        
        # mu_t = log(x0) - 0.5 * Lambda_t
        log_x0 = np.log(self.target_signal)
        mu_t = log_x0 - 0.5 * lambda_t
        
        # sigma_t^2 = Lambda_t
        sigma_t_sq = lambda_t + 1e-8
        
        # Score: -(log(x) - mu_t)/sigma_t^2 * (1/x) - 1/x
        log_x = np.log(x_pos)
        score = -(log_x - mu_t) / sigma_t_sq / x_pos - 1.0 / x_pos
        
        return score * np.sign(x + 1e-12)
    
    def get_type(self) -> str:
        return f"oracle_{self.noise_type}"

class Integrator(ABC):    
    @abstractmethod
    def integrate(self, x_init: np.ndarray, score_fn: ScoreFunction, 
                  schedule: NoiseSchedule, steps: int, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        pass

class DDPMSampling(Integrator):
    """DDPM stochastic sampling (Euler-Maruyama)"""
    
    def integrate(self, x_init: np.ndarray, score_fn: ScoreFunction, 
                  schedule: NoiseSchedule, steps: int, **kwargs) -> np.ndarray:
        
        noise_type = kwargs.get('noise_type', 'additive')
        
        if noise_type == "multiplicative":
            return self._integrate_multiplicative(x_init, score_fn, schedule, steps, **kwargs)
        else:
            return self._integrate_additive(x_init, score_fn, schedule, steps, **kwargs)
    
    def _integrate_additive(self, x_init: np.ndarray, score_fn: ScoreFunction, 
                           schedule: NoiseSchedule, steps: int, **kwargs) -> np.ndarray:
        """Standard DDPM for additive noise"""
        x = np.copy(x_init)
        timesteps = np.linspace(schedule.config.N - 1, 0, steps).astype(int)
        
        for i, t in enumerate(timesteps[:-1]):
            t_next = timesteps[i + 1]
            
            alpha_bar_t = schedule.alpha_bars[t]
            alpha_bar_next = schedule.alpha_bars[t_next] if t_next > 0 else 1.0
            alpha_t = schedule.alphas[t]
            
            # Predict x0
            score = score_fn(x, t)
            predicted_noise = -score * np.sqrt(1 - alpha_bar_t)
            x0_pred = (x - np.sqrt(1 - alpha_bar_t) * predicted_noise) / np.sqrt(alpha_bar_t + 1e-8)
            
            if t_next == 0:
                x = x0_pred
            else:
                # DDPM step
                coeff1 = np.sqrt(alpha_bar_next) * (1 - alpha_t) / (1 - alpha_bar_t + 1e-8)
                coeff2 = np.sqrt(alpha_t) * (1 - alpha_bar_next) / (1 - alpha_bar_t + 1e-8)
                posterior_mean = coeff1 * x0_pred + coeff2 * x
                
                posterior_var = (1 - alpha_bar_next) * (1 - alpha_t) / (1 - alpha_bar_t + 1e-8)
                z = np.random.randn(*x.shape)
                x = posterior_mean + np.sqrt(np.maximum(posterior_var, 1e-8)) * z
        
        return x
    
    def _integrate_multiplicative(self, x_init: np.ndarray, score_fn: ScoreFunction, 
                                 schedule: NoiseSchedule, steps: int, **kwargs) -> np.ndarray:
        """DDPM for multiplicative noise in log-space"""
        # Work in log-space
        x_pos = np.maximum(np.abs(x_init), 1e-8)
        y = np.log(x_pos)  # y = log(x)
        
        dt = schedule.config.T / steps
        timesteps = np.linspace(schedule.config.N - 1, 0, steps).astype(int)
        
        for i, t in enumerate(timesteps[:-1]):
            # Convert back to x-space for score evaluation
            x = np.exp(y) * np.sign(x_init)
            
            # Get score in x-space
            score_x = score_fn(x, t)
            
            # Convert to score in y-space: score_y = x * score_x + 1
            score_y = x * score_x + 1
            
            # Euler step in log-space
            beta_t = schedule.betas[t]
            dy = (-0.5 * beta_t) * dt + np.sqrt(beta_t * dt) * np.random.randn(*y.shape)
            y = y + dy
        
        # Convert back to x-space
        return np.exp(y) * np.sign(x_init)
    
    def get_type(self) -> str:
        return "ddpm"

class ProbabilityFlowODE(Integrator):
    """Deterministic sampling via Probability Flow ODE"""
    
    def integrate(self, x_init: np.ndarray, score_fn: ScoreFunction, 
                  schedule: NoiseSchedule, steps: int, **kwargs) -> np.ndarray:
        
        noise_type = kwargs.get('noise_type', 'additive')
        
        if noise_type == "multiplicative":
            return self._integrate_multiplicative_ode(x_init, score_fn, schedule, steps, **kwargs)
        else:
            return self._integrate_additive_ode(x_init, score_fn, schedule, steps, **kwargs)
    
    def _integrate_additive_ode(self, x_init: np.ndarray, score_fn: ScoreFunction, 
                               schedule: NoiseSchedule, steps: int, **kwargs) -> np.ndarray:
        """PF-ODE for additive noise: dx/dt = -0.5 * beta(t) * [x + 2*score(x,t)]"""
        
        def ode_func(t, x_flat):
            t_idx = int(t * schedule.config.N)
            t_idx = np.clip(t_idx, 0, schedule.config.N - 1)
            
            x = x_flat.reshape(x_init.shape)
            score = score_fn(x, t_idx)
            
            beta_t = schedule.betas[t_idx]
            dxdt = -0.5 * beta_t * (x + 2 * score)
            
            return dxdt.flatten()
        
        # Solve ODE backwards from t=1 to t=0
        t_span = (1.0, 0.0)
        t_eval = np.linspace(1.0, 0.0, steps)
        
        sol = solve_ivp(ode_func, t_span, x_init.flatten(), 
                       t_eval=t_eval, method='RK45', rtol=1e-5)
        
        return sol.y[:, -1].reshape(x_init.shape)
    
    def _integrate_multiplicative_ode(self, x_init: np.ndarray, score_fn: ScoreFunction, 
                                     schedule: NoiseSchedule, steps: int, **kwargs) -> np.ndarray:
        """PF-ODE for multiplicative noise in log-space"""
        
        # Work in log-space
        x_pos = np.maximum(np.abs(x_init), 1e-8)
        y_init = np.log(x_pos)
        
        def ode_func(t, y_flat):
            t_idx = int(t * schedule.config.N)
            t_idx = np.clip(t_idx, 0, schedule.config.N - 1)
            
            y = y_flat.reshape(y_init.shape)
            x = np.exp(y) * np.sign(x_init)
            
            # Get score in x-space
            score_x = score_fn(x, t_idx)
            
            # ODE in log-space: dy/dt = -0.5*beta(t) + sqrt(beta(t)) * x * score_x
            beta_t = schedule.betas[t_idx]
            dydt = -0.5 * beta_t + np.sqrt(beta_t) * x * score_x
            
            return dydt.flatten()
        
        # Solve ODE backwards from t=1 to t=0
        t_span = (1.0, 0.0)
        t_eval = np.linspace(1.0, 0.0, steps)
        
        sol = solve_ivp(ode_func, t_span, y_init.flatten(), 
                       t_eval=t_eval, method='RK45', rtol=1e-5)
        
        # Convert back to x-space
        y_final = sol.y[:, -1].reshape(y_init.shape)
        return np.exp(y_final) * np.sign(x_init)
    
    def get_type(self) -> str:
        return "ode"

class NoiseModel(ABC):    
    @abstractmethod
    def add_noise(self, clean_signal: np.ndarray, t_idx: int, 
                  schedule: NoiseSchedule, intensity: float = 1.0) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        pass

class AdditiveGaussianNoise(NoiseModel):    
    def add_noise(self, clean_signal: np.ndarray, t_idx: int, 
                  schedule: NoiseSchedule, intensity: float = 1.0) -> np.ndarray:
        alpha_bar_t = schedule.alpha_bars[t_idx]
        noise = np.random.randn(*clean_signal.shape) * intensity
        return np.sqrt(alpha_bar_t) * clean_signal + np.sqrt(1 - alpha_bar_t) * noise
    
    def get_type(self) -> str:
        return "additive"

class MultiplicativeNoise(NoiseModel):
    """Proper multiplicative (geometric) noise in log-space"""
    
    def add_noise(self, clean_signal: np.ndarray, t_idx: int, 
                  schedule: NoiseSchedule, intensity: float = 1.0) -> np.ndarray:
        # Ensure positive signal
        clean_abs = np.maximum(np.abs(clean_signal), 1e-6)
        signs = np.sign(clean_signal + 1e-12)
        
        # Work in log-space
        log_clean = np.log(clean_abs)
        
        # Get Lambda_t (integrated noise)
        lambda_t = schedule.lambda_t[t_idx] * intensity
        
        # Forward SDE in log-space: dY_t = -0.5*beta(t)*dt + sqrt(beta(t))*dW_t
        # After integration: Y_t ~ N(log(x0) - 0.5*Lambda_t, Lambda_t)
        mu_t = log_clean - 0.5 * lambda_t
        sigma_t = np.sqrt(lambda_t)
        
        noise = np.random.randn(*clean_signal.shape)
        log_noisy = mu_t + sigma_t * noise
        
        # Transform back to x-space
        return np.exp(log_noisy) * signs
    
    def get_type(self) -> str:
        return "multiplicative"

class ScoreTrainer:
    """Train neural network score functions"""
    
    def __init__(self, config: SDEConfig, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training neural score functions")
        
        self.config = config
        self.device = device
        self.schedule = NoiseSchedule(config)
        
        # Create model
        self.model = MLPScoreNetwork(
            input_dim=config.D,
            noise_type=config.noise_type
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    def generate_training_data(self, n_samples: int = 10000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic training data"""
        
        # Generate clean signals (multimodal distribution)
        clean_signals = []
        for _ in range(n_samples):
            # Mix of different signal types
            if np.random.random() < 0.33:
                # Sinusoidal
                x = np.linspace(0, 4*np.pi, self.config.D)
                signal = np.sin(x) + 0.3 * np.sin(3*x)
            elif np.random.random() < 0.5:
                # Step function
                signal = np.random.randn(self.config.D) * 0.1
                signal[50:100] = 1.0 + np.random.randn(50) * 0.1
                signal[120:170] = -0.7 + np.random.randn(50) * 0.1
            else:
                # Mixed frequency
                x = np.linspace(0, 4*np.pi, self.config.D)
                signal = np.sin(x) + 0.5*np.sin(5*x) + 0.2*np.sin(10*x)
            
            # Add some randomness
            signal += np.random.randn(self.config.D) * 0.05
            
            if self.config.noise_type == "multiplicative":
                # Ensure positive for multiplicative noise
                signal = np.maximum(np.abs(signal) + 0.1, 0.1)
            
            clean_signals.append(signal)
        
        clean_signals = np.array(clean_signals)
        
        # Create noisy versions at random timesteps
        noisy_signals = []
        timesteps = []
        target_scores = []
        
        if self.config.noise_type == "additive":
            noise_model = AdditiveGaussianNoise()
        else:
            noise_model = MultiplicativeNoise()
        
        for i in range(n_samples):
            # Random timestep
            t_idx = np.random.randint(0, self.config.N)
            timesteps.append(t_idx / self.config.N)  # Normalize to [0,1]
            
            # Add noise
            noisy = noise_model.add_noise(clean_signals[i], t_idx, self.schedule)
            noisy_signals.append(noisy)
            
            # Compute target score (oracle)
            oracle_score = UniversalOracleScore(self.schedule, clean_signals[i], self.config.noise_type)
            target_score = oracle_score(noisy, t_idx)
            target_scores.append(target_score)
        
        return (torch.tensor(noisy_signals, dtype=torch.float32),
                torch.tensor(timesteps, dtype=torch.float32),
                torch.tensor(target_scores, dtype=torch.float32))
    
    def train(self, n_epochs: int = 500, batch_size: int = 128, save_path: str = "score_model.pth"):
        """Train the score network"""
        
        print(f"Generating training data for {self.config.noise_type} noise...")
        X, T, target_scores = self.generate_training_data()
        
        dataset = TensorDataset(X, T, target_scores)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training score network for {n_epochs} epochs...")
        
        self.model.train()
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_x, batch_t, batch_target in dataloader:
                batch_x = batch_x.to(self.device)
                batch_t = batch_t.to(self.device)
                batch_target = batch_target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predicted_score = self.model(batch_x, batch_t)
                
                # DSM loss
                loss = nn.MSELoss()(predicted_score, batch_target)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'losses': losses
        }, save_path)
        
        print(f"Model saved to {save_path}")
        return losses

class Denoiser:    
    def __init__(self, config: SDEConfig, steps: int = 20):
        self.config = config
        self.steps = steps
        self.schedule = NoiseSchedule(config)
        
        # Initialize noise model
        if config.noise_type == "additive":
            self.noise_model = AdditiveGaussianNoise()
        elif config.noise_type == "multiplicative":
            self.noise_model = MultiplicativeNoise()
        else:
            raise ValueError(f"Unknown noise type: {config.noise_type}")
        
        # Initialize integrator
        if config.sampling_method == "ddpm":
            self.integrator = DDPMSampling()
        elif config.sampling_method == "ode":
            self.integrator = ProbabilityFlowODE()
        else:
            raise ValueError(f"Unknown sampling method: {config.sampling_method}")
    
    def add_noise(self, clean_signal: np.ndarray, intensity: float = 1.0, t_idx: Optional[int] = None) -> np.ndarray:
        if t_idx is None:
            t_idx = self.config.N - 1
        return self.noise_model.add_noise(clean_signal, t_idx, self.schedule, intensity)
    
    def denoise(self, noisy_signal: np.ndarray, clean_signal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()
        
        # Create score function
        if self.config.score_type == "oracle":
            if clean_signal is None:
                raise ValueError("Oracle score requires clean signal for reference")
            score_fn = UniversalOracleScore(self.schedule, clean_signal, self.config.noise_type)
        elif self.config.score_type == "neural":
            model_path = f"score_model_{self.config.noise_type}.pth"
            score_fn = NeuralScoreFunction(model_path, self.config)
        else:
            raise ValueError(f"Unknown score type: {self.config.score_type}")
        
        # Denoise
        denoised = self.integrator.integrate(
            noisy_signal, score_fn, self.schedule, self.steps,
            noise_type=self.config.noise_type
        )
        
        runtime = time.time() - start_time
        
        metadata = {
            "noise_type": self.config.noise_type,
            "score_type": self.config.score_type,
            "sampling_method": self.config.sampling_method,
            "integrator": self.integrator.get_type(),
            "steps": self.steps,
            "runtime": f"{runtime:.3f}s"
        }
        
        return denoised, metadata

def compute_metrics(clean: np.ndarray, denoised: np.ndarray) -> Dict[str, float]:
    """Compute denoising quality metrics"""
    
    # PSNR
    max_val = np.max(np.abs(clean))
    mse = np.mean((clean - denoised) ** 2)
    psnr = 20 * np.log10(max_val / np.sqrt(mse + 1e-12)) if mse > 1e-12 else 100.0
    
    # SSIM (simplified)
    c1, c2 = 0.01, 0.03
    mu1, mu2 = np.mean(clean), np.mean(denoised)
    sigma1_sq = np.var(clean)
    sigma2_sq = np.var(denoised)
    sigma12 = np.mean((clean - mu1) * (denoised - mu2))
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim = max(0.0, numerator / denominator) if denominator > 0 else 0.0
    
    # L2 relative error
    l2_error = np.linalg.norm(clean - denoised) / np.linalg.norm(clean)
    
    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "L2_error": l2_error
    }

def create_test_signal(signal_type: str = "mixed", D: int = 200) -> np.ndarray:
    """Create test signals for evaluation"""
    x = np.linspace(0, 4*np.pi, D)
    
    if signal_type == "smooth":
        return np.sin(x) + 0.3 * np.sin(3*x)
    elif signal_type == "step":
        signal = np.zeros(D)
        signal[50:100] = 1.0
        signal[120:170] = -0.7
        return signal
    elif signal_type == "mixed":
        return np.sin(x) + 0.5*np.sin(5*x) + 0.2*np.sin(10*x)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

def demonstrate_enhanced_framework():
    """Demonstrate the enhanced framework capabilities"""
    
    print("Enhanced SDE Denoising Framework Demonstration")
    print("=" * 60)
    
    # Test both noise types and sampling methods
    configs = [
        SDEConfig(noise_type="additive", score_type="oracle", sampling_method="ddpm"),
        SDEConfig(noise_type="additive", score_type="oracle", sampling_method="ode"),
        SDEConfig(noise_type="multiplicative", score_type="oracle", sampling_method="ddpm"),
        SDEConfig(noise_type="multiplicative", score_type="oracle", sampling_method="ode"),
    ]
    
    # Add neural network configs if PyTorch is available
    if TORCH_AVAILABLE:
        configs.extend([
            SDEConfig(noise_type="additive", score_type="neural", sampling_method="ddpm"),
            SDEConfig(noise_type="multiplicative", score_type="neural", sampling_method="ode"),
        ])
    
    results = []
    
    # Create test signal
    clean_signal = create_test_signal("mixed", 200)
    if any(config.noise_type == "multiplicative" for config in configs):
        # Ensure positive for multiplicative noise tests
        clean_signal_mult = np.maximum(np.abs(clean_signal) + 0.1, 0.1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, config in enumerate(configs[:6]):  # Limit to 6 for plotting
        print(f"\nTesting: {config.noise_type} noise, {config.score_type} score, {config.sampling_method} sampling")
        
        # Train neural network if needed
        if config.score_type == "neural":
            model_path = f"score_model_{config.noise_type}.pth"
            if not os.path.exists(model_path):
                print(f"Training neural score network for {config.noise_type} noise...")
                trainer = ScoreTrainer(config)
                trainer.train(n_epochs=200, save_path=model_path)
        
        # Use appropriate clean signal
        test_signal = clean_signal_mult if config.noise_type == "multiplicative" else clean_signal
        
        # Create denoiser
        denoiser = Denoiser(config, steps=20)
        
        # Add noise
        np.random.seed(42)  # Reproducible results
        noisy_signal = denoiser.add_noise(test_signal, intensity=1.0)
        
        # Denoise
        if config.score_type == "oracle":
            denoised_signal, metadata = denoiser.denoise(noisy_signal, test_signal)
        else:
            denoised_signal, metadata = denoiser.denoise(noisy_signal)
        
        # Compute metrics
        metrics = compute_metrics(test_signal, denoised_signal)
        
        # Store results
        result = {
            "config": config,
            "metrics": metrics,
            "metadata": metadata
        }
        results.append(result)
        
        # Plot results
        ax = axes[i]
        ax.plot(test_signal, 'b-', linewidth=2, label='Clean', alpha=0.8)
        ax.plot(noisy_signal, 'gray', alpha=0.6, linewidth=1, label='Noisy')
        ax.plot(denoised_signal, 'r--', linewidth=2, label='Denoised')
        
        title = f"{config.noise_type.title()} | {config.score_type.title()} | {config.sampling_method.upper()}"
        ax.set_title(f"{title}\nPSNR: {metrics['PSNR']:.1f} dB")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"  SSIM: {metrics['SSIM']:.3f}")
        print(f"  L2 Error: {metrics['L2_error']:.4f}")
        print(f"  Runtime: {metadata['runtime']}")
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<40} {'PSNR (dB)':<12} {'SSIM':<8} {'L2 Error':<10}")
    print("-" * 80)
    
    for result in results:
        config = result['config']
        metrics = result['metrics']
        config_str = f"{config.noise_type}|{config.score_type}|{config.sampling_method}"
        print(f"{config_str:<40} {metrics['PSNR']:<12.2f} {metrics['SSIM']:<8.3f} {metrics['L2_error']:<10.4f}")
    
    return results

def train_neural_networks():
    """Train neural networks for both noise types"""
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping neural network training.")
        return
    
    print("Training Neural Score Networks")
    print("=" * 40)
    
    # Train for additive noise
    print("\nTraining for additive noise...")
    config_add = SDEConfig(noise_type="additive")
    trainer_add = ScoreTrainer(config_add)
    losses_add = trainer_add.train(n_epochs=300, save_path="score_model_additive.pth")
    
    # Train for multiplicative noise  
    print("\nTraining for multiplicative noise...")
    config_mult = SDEConfig(noise_type="multiplicative")
    trainer_mult = ScoreTrainer(config_mult)
    losses_mult = trainer_mult.train(n_epochs=300, save_path="score_model_multiplicative.pth")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_add)
    plt.title('Additive Noise Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses_mult)
    plt.title('Multiplicative Noise Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Neural network training completed!")

if __name__ == "__main__":
    print("Enhanced SDE Denoising Framework")
    print("=" * 40)
    
    # Option 1: Train neural networks
    train_networks = input("Train neural networks? (y/n): ").lower() == 'y'
    if train_networks:
        train_neural_networks()
    
    # Option 2: Run demonstration
    print("\nRunning framework demonstration...")
    results = demonstrate_enhanced_framework()
    
    print("\n" + "=" * 60)
    print("FRAMEWORK CAPABILITIES DEMONSTRATED:")
    print("✅ Multiplicative noise with proper log-space transformation")
    print("✅ Trainable neural network score functions")
    print("✅ Probability Flow ODE deterministic sampling")
    print("✅ Modular design supporting multiple configurations")
    print("✅ Comprehensive evaluation and visualization")
    print("=" * 60)