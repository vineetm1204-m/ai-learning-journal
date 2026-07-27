import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# ============================================================
# Day 32: GAN Mini-Experiment — Generator vs Discriminator
# ============================================================

# ---- Reproducibility ----
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Hyperparameters ----
latent_dim = 100
batch_size = 128
lr = 2e-4
epochs = 10          # keep short for a journal demo
img_size = 28
channels = 1
sample_interval = 200

# ---- Data ----
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   # [-1, 1]
])
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
)

# ---- Models ----
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), channels, img_size, img_size)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        return self.model(flat)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

# ---- Loss & Optimizers ----
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# ---- Logging ----
G_losses, D_losses = [], []
os.makedirs("gan_samples", exist_ok=True)

# ---- Training Loop ----
print(f"Training on {device} | latent_dim={latent_dim} | batch_size={batch_size}")
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        # Labels
        valid = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size_curr, latent_dim, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Logging
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            save_image(gen_imgs.data[:25], f"gan_samples/{batches_done}.png", nrow=5, normalize=True)

# ---- Final Sample Grid ----
with torch.no_grad():
    z = torch.randn(25, latent_dim, device=device)
    final_imgs = generator(z).cpu()
    save_image(final_imgs, "gan_samples/final_grid.png", nrow=5, normalize=True)

# ---- Loss Curves ----
plt.figure(figsize=(8, 4))
plt.plot(G_losses, label="Generator", alpha=0.7)
plt.plot(D_losses, label="Discriminator", alpha=0.7)
plt.xlabel("Iterations")
plt.ylabel("BCE Loss")
plt.title("GAN Training Losses (Day 32)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gan_samples/loss_curve.png", dpi=150)
plt.close()

print("\nDone. Samples in ./gan_samples/")