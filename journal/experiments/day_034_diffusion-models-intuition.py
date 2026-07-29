import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
T = 1000                    # Total diffusion timesteps
IMG_SIZE = 32               # Image dimension (32x32)
N_SAMPLES = 4               # Number of sample trajectories
BETA_START = 1e-4           # Noise schedule start
BETA_END = 0.02             # Noise schedule end
VIS_TIMESTEPS = [0, 50, 100, 200, 400, 600, 800, 999]  # Timesteps to visualize

# ============================================================
# NOISE SCHEDULE (Linear beta schedule)
# ============================================================
betas = np.linspace(BETA_START, BETA_END, T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

# Pre-compute useful quantities
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

# ============================================================
# SYNTHETIC DATA: Simple geometric shapes
# ============================================================
def generate_shapes(n_samples, img_size):
    """Generate clean images with simple shapes."""
    images = np.zeros((n_samples, img_size, img_size))
    for i in range(n_samples):
        img = np.zeros((img_size, img_size))
        shape_type = np.random.choice(['circle', 'square', 'cross', 'triangle'])
        cx, cy = np.random.randint(8, img_size-8, 2)
        size = np.random.randint(4, 10)
        
        y, x = np.ogrid[:img_size, :img_size]
        if shape_type == 'circle':
            mask = (x - cx)**2 + (y - cy)**2 <= size**2
        elif shape_type == 'square':
            mask = (np.abs(x - cx) <= size) & (np.abs(y - cy) <= size)
        elif shape_type == 'cross':
            mask = (np.abs(x - cx) <= size//2) | (np.abs(y - cy) <= size//2)
        else:  # triangle
            mask = (y >= cy - size) & (y <= cy + size) & (np.abs(x - cx) <= (y - cy + size) * 0.5)
        
        img[mask] = 1.0
        # Add slight blur for realism
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=0.8)
        images[i] = img / img.max()
    return images

# ============================================================
# FORWARD PROCESS: q(x_t | x_0)
# ============================================================
def q_sample(x_0, t, noise=None):
    """
    Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    """
    if noise is None:
        noise = np.random.randn(*x_0.shape)
    sqrt_alpha_bar = sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t]
    return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

def q_posterior_mean(x_0, x_t, t):
    """Mean of q(x_{t-1} | x_t, x_0)"""
    coef1 = betas[t] * sqrt_alphas_cumprod_prev[t] / (1.0 - alphas_cumprod[t])
    coef2 = (1.0 - alphas_cumprod_prev[t]) * np.sqrt(alphas[t]) / (1.0 - alphas_cumprod[t])
    return coef1 * x_0 + coef2 * x_t

# ============================================================
# REVERSE PROCESS: p_theta(x_{t-1} | x_t) - SIMPLE PREDICTOR
# ============================================================
class SimpleDenoiser:
    """
    A tiny neural network to predict noise (epsilon) from noisy image x_t and timestep t.
    In practice this would be a U-Net. Here we use a simple CNN-like structure with numpy.
    """
    def __init__(self, img_size, hidden=64):
        self.img_size = img_size
        # Simple learned parameters (simulating a trained network)
        # In reality these would be learned; here we use a heuristic denoiser
        self.kernel_size = 3
        
    def predict_noise(self, x_t, t):
        """
        Heuristic denoiser: estimates noise by comparing local statistics.
        This mimics what a trained network would learn.
        """
        # Normalize timestep
        t_norm = t / T
        
        # Simple denoising: weighted average of neighbors (non-local means idea)
        # This is a stand-in for a learned U-Net
        pad = 1
        x_padded = np.pad(x_t, ((pad, pad), (pad, pad)), mode='reflect')
        
        # Local mean (3x3 neighborhood)
        local_mean = np.zeros_like(x_t)
        for dy in range(3):
            for dx in range(3):
                local_mean += x_padded[dy:dy+self.img_size, dx:dx+self.img_size]
        local_mean /= 9.0
        
        # Estimate noise as residual from local smoothness
        # At high noise (large t), trust local mean more
        # At low noise (small t), trust input more
        alpha = 0.3 + 0.5 * t_norm  # blending factor
        denoised = alpha * local_mean + (1 - alpha) * x_t
        
        # Predicted noise = (x_t - sqrt_alpha_bar * denoised) / sqrt_one_minus_alpha_bar
        sqrt_alpha_bar = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t]
        eps_pred = (x_t - sqrt_alpha_bar * denoised) / (sqrt_one_minus_alpha_bar + 1e-8)
        
        return eps_pred

# ============================================================
# SAMPLING (Reverse process)
# ============================================================
def p_sample(denoiser, x_t, t):
    """Single reverse step: x_{t-1} ~ p(x_{t-1} | x_t)"""
    eps_pred = denoiser.predict_noise(x_t, t)
    
    # Predict x_0 from x_t and predicted noise
    sqrt_alpha_bar = sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t]
    x_0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / (sqrt_alpha_bar + 1e-8)
    x_0_pred = np.clip(x_0_pred, 0, 1)
    
    # Posterior mean
    if t > 0:
        mean = q_posterior_mean(x_0_pred, x_t, t)
        var = posterior_variance[t]
        noise = np.random.randn(*x_t.shape) * np.sqrt(var)
        x_prev = mean + noise
    else:
        x_prev = x_0_pred
    
    return np.clip(x_prev, 0, 1), x_0_pred

def sample_chain(denoiser, x_T, save_steps=None):
    """Run full reverse process from x_T to x_0."""
    if save_steps is None:
        save_steps = VIS_TIMESTEPS
    x_t = x_T.copy()
    trajectory = {T: x_t.copy()}
    x_0_preds = {}
    
    for t in reversed(range(T)):
        x_t, x_0_pred = p_sample(denoiser, x_t, t)
        if t in save_steps:
            trajectory[t] = x_t.copy()
            x_0_preds[t] = x_0_pred.copy()
    
    trajectory[0] = x_t.copy()
    x_0_preds[0] = x_t.copy()
    return trajectory, x_0_preds

# ============================================================
# VISUALIZATION
# ============================================================
def plot_forward_process(clean_imgs, forward_trajectories, save_path='forward_process.png'):
    """Visualize forward diffusion: clean -> noise."""
    n_samples = len(clean_imgs)
    n_steps = len(VIS_TIMESTEPS)
    
    fig, axes = plt.subplots(n_samples, n_steps + 1, figsize=(2.5 * (n_steps + 1), 2.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Clean image
        axes[i, 0].imshow(clean_imgs[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Clean (t=0)', fontsize=10)
        axes[i, 0].axis('off')
        
        # Noisy versions
        for j, t in enumerate(VIS_TIMESTEPS):
            axes[i, j+1].imshow(forward_trajectories[i][t], cmap='gray', vmin=0, vmax=1)
            axes[i, j+1].set_title(f't={t}', fontsize=10)
            axes[i, j+1].axis('off')
    
    plt.suptitle('Forward Diffusion Process: Adding Noise', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved forward process to {save_path}")

def plot_reverse_process(reverse_trajectories, x_0_preds, save_path='reverse_process.png'):
    """Visualize reverse diffusion: noise -> clean."""
    n_samples = len(reverse_trajectories)
    n_steps = len(VIS_TIMESTEPS) + 1  # +1 for t=0
    
    fig, axes = plt.subplots(n_samples, n_steps, figsize=(2.5 * n_steps, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    vis_steps = VIS_TIMESTEPS + [0]
    
    for i in range(n_samples):
        for j, t in enumerate(vis_steps):
            axes[i, j].imshow(reverse_trajectories[i][t], cmap='gray', vmin=0, vmax=1)
            title = f't={t}'
            if t in x_0_preds[i]:
                title += f'\n(pred x₀)'
            axes[i, j].set_title(title, fontsize=10)
            axes[i, j].axis('off')
    
    plt.suptitle('Reverse Diffusion Process: Denoising', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reverse process to {save_path}")

def plot_noise_schedule(save_path='noise_schedule.png'):
    """Visualize the noise schedule."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Beta schedule
    axes[0, 0].plot(betas)
    axes[0, 0].set_title('β_t (Noise Schedule)')
    axes[0, 0].set_xlabel('Timestep t')
    axes[0, 0].set_ylabel('β_t')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Alpha bar
    axes[0, 1].plot(alphas_cumprod)
    axes[0, 1].set_title('ᾱ_t = ∏(1-β_s)')
    axes[0, 1].set_xlabel('Timestep t')
    axes[0, 1].set_ylabel('ᾱ_t')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Signal/Noise ratio
    snr = alphas_cumprod / (1 - alphas_cumprod)
    axes[1, 0].plot(snr)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('SNR = ᾱ_t / (1-ᾱ_t)')
    axes[1, 0].set_xlabel('Timestep t')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Standard deviation of noise added
    axes[1, 1].plot(sqrt_one_minus_alphas_cumprod)
    axes[1, 1].set_title('√(1-ᾱ_t) - Noise Std Dev')
    axes[1, 1].set_xlabel('Timestep t')
    axes[1, 1].set_ylabel('Std')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Diffusion Noise Schedule Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved noise schedule to {save_path}")

def plot_comparison(clean_imgs, forward_trajs, reverse_trajs, save_path='comparison.png'):
    """Side-by-side comparison of forward and reverse."""
    n_samples = len(clean_imgs)
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original
        axes[i, 0].imshow(clean_imgs[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Original (x₀)', fontsize=12)
        axes[i, 0].axis('off')
        
        # Fully noised (t=T-1)
        axes[i, 1].imshow(forward_trajs[i][999], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Pure Noise (x_T)', fontsize=12)
        axes[i, 1].axis('off')
        
        # Reconstructed
        axes[i, 2].imshow(reverse_trajs[i][0], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Reconstructed (x̂₀)', fontsize=12)
        axes[i, 2].axis('off')
    
    plt.suptitle('Diffusion Model: Forward → Reverse', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {save_path}")

def plot_pixel_trajectories(forward_trajs, reverse_trajs, save_path='pixel_trajectories.png'):
    """Track specific pixels through time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Pick 4 pixels from first sample
    pixels = [(8, 8), (16, 16), (24, 8), (8, 24)]
    colors = ['red', 'blue', 'green', 'orange']
    
    for idx, ((py, px), color) in enumerate(zip(pixels, colors)):
        ax = axes[idx // 2, idx % 2]
        
        # Forward
        forward_vals = [forward_trajs[0][t][py, px] for t in range(T)]
        ax.plot(range(T), forward_vals, color=color, alpha=0.7, label='Forward', linewidth=0.8)
        
        # Reverse (only saved steps)
        vis_steps = VIS_TIMESTEPS + [0]
        reverse_vals = [reverse_trajs[0][t][py, px] for t in vis_steps]
        ax.plot(vis_steps, reverse_vals, color=color, linestyle='--', label='Reverse', linewidth=1.5, marker='o', markersize=3)
        
        ax.set_title(f'Pixel ({py}, {px}) Trajectory')
        ax.set_xlabel('Timestep t')
        ax.set_ylabel('Pixel Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, T)
        ax.set_ylim(-0.1, 1.1)
    
    plt.suptitle('Pixel Value Evolution: Forward vs Reverse', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pixel trajectories to {save_path}")

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def main():
    print("=" * 60)
    print("DAY 34: DIFFUSION MODELS INTUITION - MINI EXPERIMENT")
    print("=" * 60)
    
    # 1. Generate clean data
    print("\n[1/5] Generating synthetic clean images...")
    clean_images = generate_shapes(N_SAMPLES, IMG_SIZE)
    print(f"    Generated {N_SAMPLES} images of size {IMG_SIZE}x{IMG_SIZE}")
    
    # 2. Forward process: add noise
    print("\n[2/5] Running forward diffusion (adding noise)...")
    forward_trajectories = []
    for i in range(N_SAMPLES):
        traj = {0: clean_images[i].copy()}
        x_t = clean_images[i].copy()
        for t in range(1, T):
            x_t, _ = q_sample(x_t, t)  # Note: this accumulates noise differently than closed form
            if t in VIS_TIMESTEPS:
                traj[t] = x_t.copy()
        traj[T-1] = x_t.copy()
        forward_trajectories.append(traj)
    print(f"    Completed {T} timesteps of forward diffusion")
    
    # Also compute closed-form for verification at key timesteps
    print("    Computing closed-form forward samples for visualization...")
    forward_closed_form = []
    for i in range(N_SAMPLES):
        traj = {}
        for t in VIS_TIMESTEPS:
            x_t, _ = q_sample(clean_images[i], t)
            traj[t] = x_t
        traj[0] = clean_images[i]
        traj[999] = q_sample(clean_images[i], 999)[0]
        forward_closed_form.append(traj)
    
    # 3. Initialize denoiser
    print("\n[3/5] Initializing denoiser network...")
    denoiser = SimpleDenoiser(IMG_SIZE)
    print("    Using heuristic denoiser (stand-in for trained U-Net)")
    
    # 4. Reverse process: denoise
    print("\n[4/5] Running reverse diffusion (denoising)...")
    reverse_trajectories = []
    x_0_predictions = []
    for i in range(N_SAMPLES):
        # Start from pure noise
        x_T = np.random.randn(IMG_SIZE, IMG_SIZE)
        x_T = (x_T - x_T.min()) / (x_T.max() - x_T.min())  # Normalize to [0,1]
        
        traj, x0_preds = sample_chain(denoiser, x_T)
        reverse_trajectories.append(traj)
        x_0_predictions.append(x0_preds)
    print(f"    Completed {T} timesteps of reverse diffusion")
    
    # 5. Visualizations
    print("\n[5/5] Generating visualizations...")
    plot_noise_schedule()
    plot_forward_process(clean_images, forward_closed_form)
    plot_reverse_process(reverse_trajectories, x_0_predictions)
    plot_comparison(clean_images, forward_closed_form, reverse_trajectories)
    plot_pixel_trajectories(forward_closed_form, reverse_trajectories)
    
    # Quantitative metrics
    print("\n" + "=" * 60)
    print("QUANTITATIVE RESULTS")
    print("=" * 60)
    for i in range(N_SAMPLES):
        mse = np.mean((clean_images[i] - reverse_trajectories[i][0])**2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
        print(f"  Sample {i+1}: MSE = {mse:.6f}, PSNR = {psnr:.2f} dB")
    
    avg_mse = np.mean([np.mean((clean_images[i] - reverse_trajectories[i][0])**2) for i in range(N_SAMPLES)])
    print(f"\n  Average MSE: {avg_mse:.6f}")
    print(f"  Average PSNR: {20 * np.log10(1.0 / np.sqrt(avg_mse + 1e-8)):.2f} dB")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
1. FORWARD PROCESS: Gradually destroys structure by adding Gaussian noise
   - x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
   - At t=T, x_T ≈ pure noise (ᾱ_T ≈ 0)

2. REVERSE PROCESS: Learns to predict noise ε_θ(x_t, t) to recover x₀
   - x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ) + σ_t z
   - Our heuristic denoiser approximates this without training

3. NOISE SCHEDULE: Controls the signal-to-noise ratio over time
   - Linear β schedule: slow start, faster noise addition later
   - SNR drops exponentially, making early steps easier to denoise

4. TRACTABILITY: Closed-form forward allows training on any t
   - Can sample t uniformly and compute loss: ||ε - ε_θ(x_t, t)||²
   - No need to simulate full chain during training

5. SAMPLING: Requires sequential steps (slow) but produces high-quality samples
   - Each step refines the estimate slightly
   - Modern methods (DDIM, DPM-Solver) accelerate this
""")
    print("Experiment complete! Check generated PNG files.")

if __name__ == "__main__":
    main()