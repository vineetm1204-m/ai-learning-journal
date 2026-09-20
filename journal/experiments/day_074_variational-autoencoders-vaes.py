import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 20
input_dim = 784  # 28x28
hidden_dim = 400

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialize model and optimizer
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Starting training...")
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

# Evaluation and visualization
model.eval()
with torch.no_grad():
    # Generate samples from latent space
    sample = torch.randn(64, latent_dim).to(device)
    generated = model.decode(sample).cpu()
    
    # Reconstruct test images
    test_data, _ = next(iter(test_loader))
    test_data = test_data.to(device)
    recon, _, _ = model(test_data)
    recon = recon.cpu()
    test_data = test_data.cpu()

# Plot results
fig, axes = plt.subplots(4, 16, figsize=(16, 4))
for i in range(16):
    # Original
    axes[0, i].imshow(test_data[i].view(28, 28), cmap='gray')
    axes[0, i].axis('off')
    # Reconstruction
    axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
    axes[1, i].axis('off')
    # Generated
    axes[2, i].imshow(generated[i].view(28, 28), cmap='gray')
    axes[2, i].axis('off')
    # Latent space interpolation (first two dims)
    if i < 8:
        z = torch.zeros(1, latent_dim)
        z[0, 0] = -3 + i * 0.75
        z[0, 1] = 0
        interp = model.decode(z).cpu()
        axes[3, i].imshow(interp.view(28, 28), cmap='gray')
        axes[3, i].axis('off')

axes[0, 0].set_ylabel('Original', rotation=90, size='large')
axes[1, 0].set_ylabel('Reconstructed', rotation=90, size='large')
axes[2, 0].set_ylabel('Generated', rotation=90, size='large')
axes[3, 0].set_ylabel('Interpolated', rotation=90, size='large')
plt.tight_layout()
plt.savefig('vae_results.png')
print("Results saved to vae_results.png")

# Latent space visualization (2D if latent_dim=2, else PCA)
if latent_dim == 2:
    with torch.no_grad():
        latent_points = []
        labels_list = []
        for data, labels in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, input_dim))
            latent_points.append(mu.cpu())
            labels_list.append(labels)
        latent_points = torch.cat(latent_points).numpy()
        labels_list = torch.cat(labels_list).numpy()
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_points[:, 0], latent_points[:, 1], c=labels_list, cmap='tab10', alpha=0.6, s=2)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization (2D)')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.savefig('latent_space.png')
    print("Latent space visualization saved to latent_space.png")
else:
    print(f"Latent dimension is {latent_dim}, skipping 2D visualization. Set latent_dim=2 for 2D plot.")

print("Experiment completed.")