import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
BATCH_SIZE = 128
LATENT_DIM = 100
LR = 0.0002
BETA1 = 0.5
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_DIR = "gan_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        flat = img.view(-1, 784)
        return self.model(flat)

# Initialize models
G = Generator(LATENT_DIM).to(DEVICE)
D = Discriminator().to(DEVICE)

# Loss and optimizers
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

# Fixed noise for consistent sampling
fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

# Training history
history = {'G_loss': [], 'D_loss': [], 'D_real_acc': [], 'D_fake_acc': []}

def save_samples(epoch, n_row=8):
    G.eval()
    with torch.no_grad():
        fake = G(fixed_noise).cpu()
    G.train()
    
    fig, axes = plt.subplots(n_row, n_row, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake[i, 0], cmap='gray')
        ax.axis('off')
    plt.suptitle(f"Epoch {epoch+1}")
    plt.tight_layout()
    plt.savefig(f"{SAMPLE_DIR}/epoch_{epoch+1:03d}.png")
    plt.close()

print(f"Training on {DEVICE}")
print(f"Generator params: {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

# Training loop
for epoch in range(EPOCHS):
    epoch_G_loss = 0.0
    epoch_D_loss = 0.0
    epoch_D_real_acc = 0.0
    epoch_D_fake_acc = 0.0
    num_batches = 0
    
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)
        
        # --- Train Discriminator ---
        opt_D.zero_grad()
        
        # Real images
        real_output = D(real_imgs)
        d_real_loss = criterion(real_output, real_labels)
        d_real_acc = (real_output > 0.5).float().mean().item()
        
        # Fake images
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z).detach()
        fake_output = D(fake_imgs)
        d_fake_loss = criterion(fake_output, fake_labels)
        d_fake_acc = (fake_output < 0.5).float().mean().item()
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        opt_D.step()
        
        # --- Train Generator ---
        opt_G.zero_grad()
        
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z)
        fake_output = D(fake_imgs)
        g_loss = criterion(fake_output, real_labels)
        
        g_loss.backward()
        opt_G.step()
        
        # Accumulate
        epoch_G_loss += g_loss.item()
        epoch_D_loss += d_loss.item()
        epoch_D_real_acc += d_real_acc
        epoch_D_fake_acc += d_fake_acc
        num_batches += 1
    
    # Average metrics
    avg_G = epoch_G_loss / num_batches
    avg_D = epoch_D_loss / num_batches
    avg_D_real = epoch_D_real_acc / num_batches
    avg_D_fake = epoch_D_fake_acc / num_batches
    
    history['G_loss'].append(avg_G)
    history['D_loss'].append(avg_D)
    history['D_real_acc'].append(avg_D_real)
    history['D_fake_acc'].append(avg_D_fake)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"G_loss: {avg_G:.4f} | D_loss: {avg_D:.4f} | "
          f"D_real_acc: {avg_D_real:.4f} | D_fake_acc: {avg_D_fake:.4f}")
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        save_samples(epoch)

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['G_loss'], label='Generator')
plt.plot(history['D_loss'], label='Discriminator')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Losses')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['D_real_acc'], label='D Real Acc')
plt.plot(history['D_fake_acc'], label='D Fake Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Discriminator Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAMPLE_DIR}/training_curves.png")
plt.close()

# Final generated grid
save_samples(EPOCHS - 1)
print(f"\nTraining complete. Samples saved to {SAMPLE_DIR}/")