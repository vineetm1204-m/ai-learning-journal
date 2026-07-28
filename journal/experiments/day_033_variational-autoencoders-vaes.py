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
LATENT_DIM = 20
EPOCHS = 10
LEARNING_RATE = 1e-3
DATA_DIR = "./data"
OUTPUT_DIR = "./vae_outputs_day33"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Day 33: VAE Experiment | Device: {DEVICE}")

# ==========================================
# Data Loading
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# Model Definition
# ==========================================
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1) # 28->14
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # 14->7
        self.enc_fc = nn.Linear(64 * 7 * 7, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 512)
        self.dec_unflatten = nn.Linear(512, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1) # 7->14
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1) # 14->28

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.view(h.size(0), -1)
        h = F.relu(self.enc_fc(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = F.relu(self.dec_unflatten(h))
        h = h.view(h.size(0), 64, 7, 7)
        h = F.relu(self.dec_conv1(h))
        return torch.sigmoid(self.dec_conv2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ==========================================
# Loss Function
# ==========================================
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE.item(), KLD.item()

# ==========================================
# Training & Evaluation Loops
# ==========================================
model = VAE(LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.4f} '
                  f'(BCE: {bce/len(data):.4f}, KLD: {kld/len(data):.4f})')
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(DEVICE)
            recon_batch, mu, logvar = model(data)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            
            # Save reconstruction comparison for first batch
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 
                           os.path.join(OUTPUT_DIR, f'reconstruction_epoch_{epoch}.png'), 
                           nrow=n)
    avg_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_loss:.4f}')
    return avg_loss

# ==========================================
# Visualization Helpers
# ==========================================
def plot_latent_space(epoch):
    model.eval()
    with torch.no_grad():
        # 1. Sample from prior (Standard Normal) -> Generate digits
        sample = torch.randn(64, LATENT_DIM).to(DEVICE)
        generated = model.decode(sample).cpu()
        save_image(generated.view(64, 1, 28, 28), 
                   os.path.join(OUTPUT_DIR, f'generation_epoch_{epoch}.png'), nrow=8)
        
        # 2. Latent Space Traversal (if 2D latent dim, else skip grid traversal)
        if LATENT_DIM == 2:
            # Create a grid in latent space
            n = 15
            grid_x = np.linspace(-3, 3, n)
            grid_y = np.linspace(-3, 3, n)
            canvas = np.empty((28 * n, 28 * n))
            for i, yi in enumerate(grid_x):
                for j, xi in enumerate(grid_y):
                    z = torch.tensor([[xi, yi]], dtype=torch.float).to(DEVICE)
                    x_decoded = model.decode(z).cpu().squeeze().numpy()
                    canvas[(n - 1 - i) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_decoded
            
            plt.figure(figsize=(10, 10))
            plt.imshow(canvas, cmap='gray')
            plt.title(f"Latent Space Manifold (Epoch {epoch})")
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR, f'manifold_epoch_{epoch}.png'))
            plt.close()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    history = {'train': [], 'test': []}
    
    for epoch in range(1, EPOCHS + 1):
        t_loss = train(epoch)
        v_loss = test(epoch)
        history['train'].append(t_loss)
        history['test'].append(v_loss)
        plot_latent_space(epoch)
    
    # Plot Loss Curve
    plt.figure()
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['test'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss (Day 33)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
    plt.close()
    
    print(f"\nExperiment complete. Outputs saved to: {OUTPUT_DIR}")