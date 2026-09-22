import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# ============================================================
# Day 75: Diffusion Models Intuition - "From Noise to Signal"
# ============================================================
# This script visualizes the core intuition of Diffusion Models:
# 1. Forward Process (Adding Noise): Gradually destroying structure (Data -> Noise).
# 2. Reverse Process (Denoising): Gradually recovering structure (Noise -> Data).
# We simulate this on a 2D "Swiss Roll" manifold to show how the model
# learns the *score function* (gradient of log probability) to reverse diffusion.

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
N_SAMPLES = 2000
N_STEPS = 50          # Diffusion timesteps
BETA_START = 0.0001
BETA_END = 0.02
SEED = 42
FIG_SIZE = (14, 6)
DPI = 100

np.random.seed(SEED)

# ------------------------------------------------------------
# 1. Data Generation: Swiss Roll (A non-trivial 2D manifold)
# ------------------------------------------------------------
def generate_swiss_roll(n_samples):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    # Add slight thickness/noise to make it a distribution
    x += 0.1 * np.random.randn(n_samples)
    y += 0.1 * np.random.randn(n_samples)
    return np.stack([x, y], axis=1)

data = generate_swiss_roll(N_SAMPLES)
# Normalize for stable visualization
data = (data - data.mean(axis=0)) / data.std(axis=0)

# ------------------------------------------------------------
# 2. Forward Process Definition (Variance Preserving SDE / DDPM)
# ------------------------------------------------------------
betas = np.linspace(BETA_START, BETA_END, N_STEPS)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

def q_sample(x_start, t, noise=None):
    """Diffuse data to step t: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps"""
    if noise is None:
        noise = np.random.randn(*x_start.shape)
    # Broadcasting for batch of t indices
    sqrt_ac = sqrt_alphas_cumprod[t][:, None]
    sqrt_om_ac = sqrt_one_minus_alphas_cumprod[t][:, None]
    return sqrt_ac * x_start + sqrt_om_ac * noise

# ------------------------------------------------------------
# 3. "Oracle" Score Function (Ground Truth for Intuition)
# ------------------------------------------------------------
# In reality, a Neural Net predicts this. Here we compute the *exact* score
# of the noisy distribution p(x_t | x_0) averaged over the data distribution.
# Score = grad_x log p(x_t) = -(x_t - sqrt(alpha_bar)x_0) / (1 - alpha_bar)
# Since we don't know x_0 for a given x_t in practice, the model learns to predict the noise (eps).
# Here we visualize the *Vector Field* of the true score at various timesteps.

def compute_true_score_field(x_grid, y_grid, t_idx):
    """Computes the true score (gradient of log prob) on a grid for a specific timestep."""
    # p(x_t) = sum_i p(x_t | x_0_i) * p(x_0_i)  (Mixture of Gaussians)
    # Score = (sum_i w_i * score_i) / (sum_i w_i)
    # where score_i = -(x_t - mu_i) / var_i
    
    mu_t = sqrt_alphas_cumprod[t_idx] * data # Means of components (N_SAMPLES, 2)
    var_t = 1.0 - alphas_cumprod[t_idx]      # Variance (scalar)
    
    # Grid points (G, 2)
    grid_pts = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    
    # Diff: (G, N, 2)
    diff = grid_pts[:, None, :] - mu_t[None, :, :]
    
    # Log weights: -0.5 * ||diff||^2 / var_t
    dist_sq = np.sum(diff**2, axis=2)
    log_weights = -0.5 * dist_sq / var_t
    
    # Stability: subtract max
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights) # (G, N)
    
    # Weighted average of scores
    # Score component i = -diff_i / var_t
    scores = -diff / var_t # (G, N, 2)
    
    # Numerator: sum(w * score)
    num = np.sum(weights[:, :, None] * scores, axis=1)
    # Denominator: sum(w)
    den = np.sum(weights, axis=1)[:, None] + 1e-8
    
    return (num / den).reshape(x_grid.shape + (2,))

# ------------------------------------------------------------
# 4. Reverse Process Simulation (Using True Score / "Perfect Model")
# ------------------------------------------------------------
# We simulate the reverse SDE: dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dW
# For DDPM (VP SDE): f = -0.5 * beta * x, g^2 = beta
# Reverse drift: -0.5 * beta * x - beta * score
# Since score = -eps_pred / sqrt(1-alpha_bar), and eps_pred approx noise...
# We use the standard DDPM sampling update (Algorithm 2) for simplicity:
# x_{t-1} = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * eps_pred) + sqrt(beta) * z

def reverse_step_ddpm(x_t, t_idx, eps_pred):
    """Single DDPM reverse step."""
    alpha = alphas[t_idx]
    alpha_bar = alphas_cumprod[t_idx]
    beta = betas[t_idx]
    
    coeff1 = 1.0 / np.sqrt(alpha)
    coeff2 = beta / np.sqrt(1.0 - alpha_bar)
    
    mean = coeff1 * (x_t - coeff2 * eps_pred)
    
    if t_idx > 0:
        noise = np.random.randn(*x_t.shape)
        var = beta # Simplified variance (could use tilde_beta)
        return mean + np.sqrt(var) * noise
    else:
        return mean

# ------------------------------------------------------------
# 5. Visualization Setup
# ------------------------------------------------------------
# Create grid for vector field
x_min, x_max = -3.5, 3.5
y_min, y_max = -3.5, 3.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))

# Precompute Forward Trajectories (for animation frames)
# We track a fixed batch of samples through time
fixed_noise = np.random.randn(N_SAMPLES, 2)
forward_trajectory = [data]
x_t = data
for i in range(N_STEPS):
    x_t = q_sample(x_t, np.array([i]*N_SAMPLES), noise=np.random.randn(N_SAMPLES, 2))
    forward_trajectory.append(x_t)

# Precompute Reverse Trajectory (using Oracle Noise Prediction)
# eps_pred = (x_t - sqrt(alpha_bar)*x_0) / sqrt(1-alpha_bar) -> This is the TRUE noise added.
# We simulate "Perfect Model" by using the noise that *was* added in forward pass 
# (requires storing it) OR by using the True Score Field.
# Let's use the True Score Field converted to eps_pred for the reverse pass starting from pure noise.
print("Computing Reverse Trajectory (Oracle)...")
reverse_trajectory = []
x_t = np.random.randn(N_SAMPLES, 2) # Start from pure noise t=T
reverse_trajectory.append(x_t.copy())

for i in reversed(range(N_STEPS)):
    t_idx = i
    # Oracle: Predict noise using true posterior mean formula derivation
    # eps_pred = - sqrt(1-alpha_bar) * score(x_t, t)
    score_field = compute_true_score_field(x_t[:,0], x_t[:,1], t_idx) # This is slow per sample
    # Optimization: Compute score on grid and interpolate? 
    # For 2000 samples * 50 steps, direct computation is okay for a script.
    # Actually, let's just use the analytical posterior mean for the specific x_0 that generated x_t?
    # No, we don't have x_0 mapping here. 
    # SIMPLIFICATION FOR DEMO: We will visualize the *Vector Field* evolving, 
    # and show a few particles moving via the True Score (Langevin Dynamics style).
    
    # Let's do Langevin Dynamics on the True Score for the reverse viz (more stable visually)
    # dx = score * dt + sqrt(2*dt) * dW (Score-based generative modeling SDE)
    # Discretized: x_{t-1} = x_t + step_size * score(x_t, t) + sqrt(2*step_size) * noise
    # step_size relates to beta.
    
    # Use the True Score Field computed on grid, interpolate for particles.
    # To keep it self-contained and fast, we'll compute score for the specific particle positions.
    
    # Compute score for current particles
    # Vectorized computation for all particles at once:
    diff = x_t[:, None, :] - (sqrt_alphas_cumprod[t_idx] * data)[None, :, :] # (N, N_data, 2)
    dist_sq = np.sum(diff**2, axis=2)
    var_t = 1.0 - alphas_cumprod[t_idx]
    log_w = -0.5 * dist_sq / var_t
    log_w -= np.max(log_w, axis=1, keepdims=True)
    w = np.exp(log_w)
    scores = -diff / var_t
    num = np.sum(w[:, :, None] * scores, axis=1)
    den = np.sum(w, axis=1)[:, None] + 1e-8
    score_vals = num / den
    
    # Langevin Step (Reverse SDE discretization)
    # dt approx beta_t
    dt = betas[t_idx] 
    x_t = x_t + dt * score_vals + np.sqrt(2 * dt) * np.random.randn(*x_t.shape)
    reverse_trajectory.append(x_t.copy())

reverse_trajectory = reverse_trajectory[::-1] # Align t=0...T

# ------------------------------------------------------------
# 6. Animation
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI)
ax_fwd, ax_rev = axes

# Colormap for time
cmap = plt.cm.viridis
norm = Normalize(vmin=0, vmax=N_STEPS)

def init_plot():
    ax_fwd.set_title("Forward Process: Data $\\to$ Noise\n(Destroying Structure)", fontsize=12, fontweight='bold')
    ax_rev.set_title("Reverse Process: Noise $\\to$ Data\n(Recovering Structure via Score)", fontsize=12, fontweight='bold')
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)
    return []

def update(frame):
    ax_fwd.clear()
    ax_rev.clear()
    
    t = frame
    
    # --- Forward Plot ---
    # Show data at step t
    pts_fwd = forward_trajectory[t]
    colors_fwd = cmap(norm(t))
    ax_fwd.scatter(pts_fwd[:, 0], pts_fwd[:, 1], c=[colors_fwd], s=5, alpha=0.6, edgecolors='none')
    
    # Overlay Vector Field (Score) at step t
    # Score points TOWARDS high density (data manifold)
    score_fwd = compute_true_score_field(xx, yy, t)
    # Skip every other arrow for clarity
    skip = 2
    ax_fwd.quiver(xx[::skip, ::skip], yy[::skip, ::skip], 
                  score_fwd[::skip, ::skip, 0], score_fwd[::skip, ::skip, 1],
                  color='red', alpha=0.5, scale=30, width=0.003, headwidth=3, label='Score $\\nabla \\log p(x_t)$')
    
    ax_fwd.set_title(f"Forward: $t={t}/{N_STEPS}$ ($\beta_t={betas[t]:.4f}$)\nData $\\to$ Gaussian Noise", fontsize=10)
    ax_fwd.set_xlim(x_min, x_max); ax_fwd.set_ylim(y_min, y_max)
    ax_fwd.set_aspect('equal'); ax_fwd.grid(True, alpha=0.2)
    ax_fwd.legend(loc='upper right', fontsize=8)

    # --- Reverse Plot ---
    # Show reverse trajectory at step t (index t corresponds to time t)
    pts_rev = reverse_trajectory[t]
    colors_rev = cmap(norm(N_STEPS - t)) # Reverse color map
    ax_rev.scatter(pts_rev[:, 0], pts_rev[:, 1], c=[colors_rev], s=5, alpha=0.6, edgecolors='none')
    
    # Overlay Vector Field at step t
    score_rev = compute_true_score_field(xx, yy, t)
    ax_rev.quiver(xx[::skip, ::skip], yy[::skip, ::skip], 
                  score_rev[::skip, ::skip, 0], score_rev[::skip, ::skip, 1],
                  color='cyan', alpha=0.7, scale=30, width=0.003, headwidth=3, label='Score $\\nabla \\log p(x_t)$')
    
    ax_rev.set_title(f"Reverse: $t={N_STEPS-t}/{N_STEPS}$ (Denoising)\nNoise $\\to$ Data Manifold", fontsize=10)
    ax_rev.set_xlim(x_min, x_max); ax_rev.set_ylim(y_min, y_max)
    ax_rev.set_aspect('equal'); ax_rev.grid(True, alpha=0.2)
    ax_rev.legend(loc='upper right', fontsize=8)
    
    fig.suptitle("Day 75: Diffusion Models Intuition — Score-Based Generative Modeling", fontsize=14, y=1.02)
    plt.tight_layout()
    return []

print("Rendering animation... (Close window to exit)")
ani = animation.FuncAnimation(fig, update, frames=N_STEPS+1, init_func=init_plot, blit=False, interval=200, repeat=True)
plt.show()

# ------------------------------------------------------------
# 7. Static Summary Plot (Final State Comparison)
# ------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), dpi=DPI)
titles = ["Clean Data ($t=0$)", f"Noisy ($t={N_STEPS//2}$)", f"Pure Noise ($t={N_STEPS}$)"]
indices = [0, N_STEPS//2, N_STEPS]

for idx, (ax, title, t) in enumerate(zip(axes2, titles, indices)):
    pts = forward_trajectory[t]
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, c=cmap(norm(t)))
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, alpha=0.2)

fig2.suptitle("Forward Diffusion Schedule: Progressive Gaussian Corruption", fontsize=14)
plt.tight_layout()
plt.show()

print("\n--- Experiment Complete ---")
print("Key Intuition:")
print("1. Forward: Adds Gaussian noise slowly (Markov chain). Structure dissolves.")
print("2. Score Function: At any t, $\\nabla_x \\log p(x_t)$ points toward the data manifold.")
print("3. Reverse: We learn to estimate this Score (via Noise Prediction $\\epsilon_\\theta$).")
print("4. Sampling: Start from $x_T \\sim \\mathcal{N}(0,I)$, follow Score field back to $x_0$.")