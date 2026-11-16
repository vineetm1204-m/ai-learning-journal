import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Diffusion hyperparameters
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 4

# Data
N_SAMPLES = 5000
DATA_DIM = 2

# ============================================================
# TOY DATASETS
# ============================================================
def make_swiss_roll(n_samples):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1)
    data = data / 10.0
    return data

def make_two_moons(n_samples):
    from sklearn.datasets import make_moons
    data, _ = make_moons(n_samples=n_samples, noise=0.05)
    data = data * 2.0 - 1.0
    return data

def make_circles(n_samples):
    from sklearn.datasets import make_circles
    data, _ = make_circles(n_samples=n_samples, noise=0.03, factor=0.5)
    data = data * 3.0
    return data

def make_checkerboard(n_samples):
    x1 = np.random.rand(n_samples) * 4 - 2
    x2 = np.random.rand(n_samples) * 4 - 2
    mask = ((np.floor(x1) + np.floor(x2)) % 2 == 0)
    data = np.stack([x1[mask], x2[mask]], axis=1)
    while len(data) < n_samples:
        extra = make_checkerboard(n_samples - len(data))
        data = np.vstack([data, extra])
    return data[:n_samples]

DATASETS = {
    "swiss_roll": make_swiss_roll,
    "two_moons": make_two_moons,
    "circles": make_circles,
    "checkerboard": make_checkerboard,
}

# ============================================================
# DIFFUSION SCHEDULE
# ============================================================
class DiffusionSchedule:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x, t, clip_denoised=True):
        betas_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, return_all=False):
        x = torch.randn(shape, device=DEVICE)
        all_x = [x.cpu()]
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=DEVICE, dtype=torch.long)
            x = self.p_sample(model, x, t)
            if return_all:
                all_x.append(x.cpu())
        return (x, all_x) if return_all else x

# ============================================================
# NOISE PREDICTION MODEL (MLP with time embedding)
# ============================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class NoisePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim, num_layers):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = []
        layers.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = torch.cat([x, t_emb], dim=-1)
        return self.net(x)

# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, schedule, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0
    for x_batch in dataloader:
        x_batch = x_batch.to(DEVICE)
        batch_size = x_batch.shape[0]

        t = torch.randint(0, schedule.timesteps, (batch_size,), device=DEVICE).long()
        noise = torch.randn_like(x_batch)
        x_noisy = schedule.q_sample(x_batch, t, noise)

        predicted_noise = model(x_noisy, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
    return total_loss / len(dataloader.dataset)

# ============================================================
# VISUALIZATION
# ============================================================
def plot_forward_process(schedule, data, save_path=None):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    timesteps_to_show = [0, 100, 250, 400, 550, 700, 800, 900, 950, 999]

    for idx, t_val in enumerate(timesteps_to_show):
        t = torch.tensor([t_val] * len(data), device=DEVICE)
        x_t = schedule.q_sample(data, t)
        ax = axes[idx]
        ax.scatter(x_t[:, 0].cpu(), x_t[:, 1].cpu(), s=1, alpha=0.6, c=t_val, cmap='viridis')
        ax.set_title(f"t = {t_val}")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle("Forward Diffusion Process: Data $\\to$ Noise", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_reverse_process(schedule, model, shape, save_path=None):
    model.eval()
    with torch.no_grad():
        x = torch.randn(shape, device=DEVICE)
        trajectory = [x.cpu().numpy()]

        for i in reversed(range(schedule.timesteps)):
            t = torch.full((shape[0],), i, device=DEVICE, dtype=torch.long)
            x = schedule.p_sample(model, x, t)
            if i % 100 == 0 or i < 10:
                trajectory.append(x.cpu().numpy())

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    steps_to_show = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1]
    labels = ["t=999", "t=899", "t=799", "t=699", "t=599", "t=499", "t=399", "t=299", "t=199", "t=0"]

    for idx, (step, label) in enumerate(zip(steps_to_show, labels)):
        ax = axes[idx]
        data = trajectory[step]
        ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.6, c=idx, cmap='plasma')
        ax.set_title(label)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle("Reverse Diffusion Process: Noise $\\to$ Data", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curve(losses, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_comparison(original, generated, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(original[:, 0], original[:, 1], s=1, alpha=0.6, c='blue')
    axes[0].set_title("Original Data")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    axes[1].scatter(generated[:, 0], generated[:, 1], s=1, alpha=0.6, c='red')
    axes[1].set_title("Generated Samples")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].axis('off')

    plt.suptitle("Data Distribution Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(dataset_name="swiss_roll"):
    print(f"\n{'='*60}")
    print(f"Day 116: Diffusion Models Intuition")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Prepare data
    data_np = DATASETS[dataset_name](N_SAMPLES)
    data = torch.tensor(data_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Schedule
    schedule = DiffusionSchedule(TIMESTEPS, BETA_START, BETA_END, DEVICE)

    # Model
    model = NoisePredictor(DATA_DIM, HIDDEN_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Visualize forward process
    print("Visualizing forward diffusion process...")
    plot_forward_process(schedule, data.to(DEVICE), save_path=f"forward_{dataset_name}.png")

    # Training
    print(f"\nTraining for {EPOCHS} epochs...")
    losses = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, schedule, dataloader, optimizer, epoch)
        losses.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.6f}")

    plot_training_curve(losses, save_path=f"loss_{dataset_name}.png")

    # Generate samples
    print("\nGenerating samples via reverse diffusion...")
    model.eval()
    with torch.no_grad():
        generated, trajectory = schedule.p_sample_loop(model, (2000, DATA_DIM), return_all=True)

    plot_reverse_process(schedule, model, (2000, DATA_DIM), save_path=f"reverse_{dataset_name}.png")
    plot_comparison(data_np, generated.cpu().numpy(), save_path=f"comparison_{dataset_name}.png")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'schedule_state': {
            'betas': schedule.betas.cpu(),
            'alphas_cumprod': schedule.alphas_cumprod.cpu(),
        },
        'config': {
            'dataset': dataset_name,
            'timesteps': TIMESTEPS,
            'hidden_dim': HIDDEN_DIM,
        }
    }, f"diffusion_{dataset_name}.pt")

    print(f"\nExperiment complete! Artifacts saved:")
    print(f"  - forward_{dataset_name}.png")
    print(f"  - reverse_{dataset_name}.png")
    print(f"  - comparison_{dataset_name}.png")
    print(f"  - loss_{dataset_name}.png")
    print(f"  - diffusion_{dataset_name}.pt")

    return model, schedule, losses

if __name__ == "__main__":
    for ds in ["swiss_roll", "two_moons", "circles", "checkerboard"]:
        run_experiment(ds)
        print()