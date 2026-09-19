import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# ============================================================
# Day 73: GAN Mini-Experiment — Generator vs Discriminator
# Self-contained: synthetic 2D data, no external dependencies
# ============================================================

# ---------- Hyperparameters ----------
latent_dim = 2
hidden_dim = 128
lr = 2e-4
batch_size = 256
epochs = 200
sample_interval = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ---------- Synthetic Target Distribution: 8 Gaussians in a circle ----------
def sample_real(batch_size):
    centers = [(2*np.cos(2*np.pi*i/8), 2*np.sin(2*np.pi*i/8)) for i in range(8)]
    chosen = np.random.choice(8, batch_size)
    noise = np.random.randn(batch_size, 2) * 0.05
    data = np.array([centers[c] for c in chosen]) + noise
    return torch.FloatTensor(data).to(device)

# ---------- Models ----------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

G = Generator().to(device)
D = Discriminator().to(device)

# ---------- Optimizers & Loss ----------
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# ---------- Storage for Visualization ----------
history = {"G_loss": [], "D_loss": [], "samples": []}

# ---------- Training Loop ----------
print(f"Training on {device}...")
for epoch in range(1, epochs + 1):
    # ---- Train Discriminator ----
    real = sample_real(batch_size)
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z).detach()

    label_real = torch.ones(batch_size, 1, device=device)
    label_fake = torch.zeros(batch_size, 1, device=device)

    D_real = D(real)
    D_fake = D(fake)
    loss_D = criterion(D_real, label_real) + criterion(D_fake, label_fake)

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # ---- Train Generator ----
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)
    D_fake = D(fake)
    loss_G = criterion(D_fake, label_real)  # generator wants D to say "real"

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    history["G_loss"].append(loss_G.item())
    history["D_loss"].append(loss_D.item())

    # ---- Periodic Sampling ----
    if epoch % sample_interval == 0 or epoch == 1:
        with torch.no_grad():
            z_vis = torch.randn(1000, latent_dim, device=device)
            samples = G(z_vis).cpu().numpy()
        history["samples"].append((epoch, samples))
        print(f"Epoch {epoch:3d} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")

# ---------- Visualization ----------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# 1. Loss curves
ax = axes[0]
ax.plot(history["D_loss"], label="Discriminator", alpha=0.8)
ax.plot(history["G_loss"], label="Generator", alpha=0.8)
ax.set_xlabel("Iteration")
ax.set_ylabel("BCE Loss")
ax.set_title("Training Losses")
ax.legend()
ax.grid(True, alpha=0.3)

# 2-6. Generated distribution evolution
real_data = sample_real(2000).cpu().numpy()
for idx, (ep, samples) in enumerate(history["samples"]):
    if idx >= 5:
        break
    ax = axes[idx + 1]
    ax.scatter(real_data[:, 0], real_data[:, 1], c="#1f77b4", s=5, alpha=0.5, label="Real")
    ax.scatter(samples[:, 0], samples[:, 1], c="#ff7f0e", s=5, alpha=0.5, label="Generated")
    ax.set_title(f"Epoch {ep}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gan_day73_results.png", dpi=150)
print("Saved figure to gan_day73_results.png")

# ---------- Optional: Animation of Generator Evolution ----------
def make_animation():
    fig, ax = plt.subplots(figsize=(5, 5))
    real_data = sample_real(2000).cpu().numpy()
    scat_real = ax.scatter(real_data[:, 0], real_data[:, 1], c="#1f77b4", s=5, alpha=0.5, label="Real")
    scat_fake = ax.scatter([], [], c="#ff7f0e", s=5, alpha=0.5, label="Generated")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")

    def update(frame):
        ep, samples = history["samples"][frame]
        scat_fake.set_offsets(samples)
        title.set_text(f"Epoch {ep}")
        return scat_fake, title

    anim = FuncAnimation(fig, update, frames=len(history["samples"]), interval=300, blit=True)
    anim.save("gan_evolution.gif", writer="pillow", fps=3)
    print("Saved animation to gan_evolution.gif")

try:
    make_animation()
except Exception as e:
    print(f"Animation skipped: {e}")

print("Done.")