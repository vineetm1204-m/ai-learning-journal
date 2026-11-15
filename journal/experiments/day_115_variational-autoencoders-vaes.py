import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LATENT_DIM = 2          # 2D for easy visualization
EPOCHS = 15
LR = 1e-3
DATA_DIR = "./data"
OUTPUT_DIR = "./vae_outputs_day115"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Data Loading (MNIST)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    # Binarize for Bernoulli likelihood (BCE) stability, or keep continuous for MSE/Gaussian
    # Using continuous [0,1] with BCEWithLogitsLoss implies Bernoulli likelihood.
    # Using MSE implies Gaussian likelihood with fixed variance.
])
train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# Model Definition: Convolutional VAE
# ==========================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 1x28x28 -> 256x7x7 -> Flatten -> FC -> mu, logvar
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # 7x7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.enc_fc = nn.Linear(256 * 7 * 7, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder: Latent -> FC -> 256x7x7 -> Deconv -> 1x28x28
        self.dec_fc = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, 256 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1), # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1), # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), # 28x28
            nn.Sigmoid() # Output [0,1] for BCE
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.enc_fc(h), 0.2)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.leaky_relu(self.dec_fc(z), 0.2)
        h = F.leaky_relu(self.dec_fc2(h), 0.2)
        h = h.view(h.size(0), 256, 7, 7)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ==========================================
# Loss Function: ELBO (Reconstruction + KL)
# ==========================================
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction Loss: Binary Cross Entropy (Bernoulli Likelihood)
    # Input x is [0,1], recon_x is [0,1] (Sigmoid output)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence: D_KL(Q(z|x) || P(z)) ; P(z) = N(0, I)
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD, BCE.item(), KLD.item()

# ==========================================
# Training & Evaluation Loops
# ==========================================
model = ConvVAE(LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

print(f"Day 115: VAE Experiment on {DEVICE}")
print(f"Latent Dim: {LATENT_DIM}, Beta: 1.0 (Standard VAE)")

fixed_test_batch, _ = next(iter(test_loader))
fixed_test_batch = fixed_test_batch.to(DEVICE)

history = {'train_loss': [], 'test_loss': [], 'bce': [], 'kld': []}

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    history['train_loss'].append(avg_train_loss)

    # Test / Validation
    model.eval()
    test_loss = 0
    total_bce = 0
    total_kld = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(DEVICE)
            recon, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon, data, mu, logvar)
            test_loss += loss.item()
            total_bce += bce
            total_kld += kld
            
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_bce = total_bce / len(test_loader.dataset)
    avg_kld = total_kld / len(test_loader.dataset)
    
    history['test_loss'].append(avg_test_loss)
    history['bce'].append(avg_bce)
    history['kld'].append(avg_kld)
    
    scheduler.step(avg_test_loss)
    
    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} "
          f"(BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f})")

    # Save Reconstructions & Samples every epoch
    with torch.no_grad():
        # 1. Reconstructions
        recon_fixed, _, _ = model(fixed_test_batch)
        comparison = torch.cat([fixed_test_batch[:8], recon_fixed[:8]])
        save_image(comparison.cpu(), f"{OUTPUT_DIR}/recon_epoch_{epoch:03d}.png", nrow=8)
        
        # 2. Random Samples (Prior p(z))
        z = torch.randn(64, LATENT_DIM).to(DEVICE)
        samples = model.decode(z)
        save_image(samples.cpu(), f"{OUTPUT_DIR}/sample_epoch_{epoch:03d}.png", nrow=8)

# ==========================================
# Post-Training Analysis & Visualization
# ==========================================
print("\nGenerating final analysis plots...")

# 1. Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train ELBO')
plt.plot(history['test_loss'], label='Test ELBO')
plt.plot(history['bce'], label='Test BCE (Recon)', linestyle='--')
plt.plot(history['kld'], label='Test KLD (Reg)', linestyle=':')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.title('Day 115: VAE Training Dynamics')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig(f"{OUTPUT_DIR}/loss_curves.png", dpi=150)
plt.close()

# 2. Latent Space Visualization (2D Scatter colored by label)
model.eval()
latent_points = []
labels_list = []
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(DEVICE)
        mu, _ = model.encode(data)
        latent_points.append(mu.cpu().numpy())
        labels_list.append(target.numpy())

latent_points = np.concatenate(latent_points, axis=0)
labels_list = np.concatenate(labels_list, axis=0)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(latent_points[:, 0], latent_points[:, 1], c=labels_list, cmap='tab10', s=2, alpha=0.6)
plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('Latent Space Structure (Test Set Encoder Mean)')
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/latent_scatter.png", dpi=150)
plt.close()

# 3. Latent Manifold Traversal (Generating a grid)
# Only works nicely if Latent Dim == 2
if LATENT_DIM == 2:
    n = 20 # 20x20 grid
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Linearly spaced coordinates on the unit square transformed through inverse CDF (ppf) of Gaussian
    # to cover the latent space where density is high.
    from scipy.stats import norm
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    model.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(DEVICE)
                x_decoded = model.decode(z_sample)
                digit = x_decoded[0].cpu().view(digit_size, digit_size).numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Latent Manifold Traversal (Prior Grid)')
    plt.axis('off')
    plt.savefig(f"{OUTPUT_DIR}/latent_manifold.png", dpi=150)
    plt.close()

# 4. Interpolation between two digits
def interpolate_latents(model, x1, x2, steps=10):
    model.eval()
    with torch.no_grad():
        mu1, _ = model.encode(x1.unsqueeze(0).to(DEVICE))
        mu2, _ = model.encode(x2.unsqueeze(0).to(DEVICE))
        alphas = torch.linspace(0, 1, steps).to(DEVICE)
        interpolated = []
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decode(z)
            interpolated.append(recon.cpu())
        return torch.cat(interpolated, dim=0)

# Pick a '0' and a '1' from test set
idx_0 = (test_dataset.targets == 0).nonzero()[0].item()
idx_1 = (test_dataset.targets == 1).nonzero()[0].item()
img_0 = test_dataset[idx_0][0]
img_1 = test_dataset[idx_1][0]

interp_imgs = interpolate_latents(model, img_0, img_1, steps=10)
# Add originals at ends for context
grid = torch.cat([img_0.unsqueeze(0), interp_imgs, img_1.unsqueeze(0)], dim=0)
save_image(grid, f"{OUTPUT_DIR}/interpolation_0_to_1.png", nrow=12)

print(f"\nExperiment complete. Outputs saved to: {OUTPUT_DIR}")