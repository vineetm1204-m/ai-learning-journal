import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
INITIAL_LR = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "."
PLOT_PATH = os.path.join(OUTPUT_DIR, "day90_lr_schedules_comparison.png")

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Synthetic Dataset ---
def get_data_loaders(n_samples=2000, input_dim=20, n_classes=5):
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

# --- Model ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# --- Training Loop ---
def train_one_schedule(scheduler_name, scheduler_factory, train_loader):
    model = SimpleMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=1e-4)
    scheduler = scheduler_factory(optimizer)

    history = {"lr": [], "loss": []}

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        # Step scheduler (usually per epoch for these types)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / len(train_loader.dataset)
        history["lr"].append(current_lr)
        history["loss"].append(avg_loss)

    return history

# --- Scheduler Factories ---
def make_step_decay(optimizer):
    # Decay by 0.1 every 15 epochs
    return optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

def make_cosine_annealing(optimizer):
    # Single cosine cycle over EPOCHS
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

def make_warm_restarts(optimizer):
    # Restart every 10 epochs, increasing period by factor of 2
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Running on {DEVICE}...")
    train_loader = get_data_loaders()

    schedules = {
        "Step Decay (γ=0.1, step=15)": make_step_decay,
        "Cosine Annealing": make_cosine_annealing,
        "Cosine Warm Restarts (T0=10)": make_warm_restarts,
    }

    results = {}
    for name, factory in schedules.items():
        print(f"\nTraining: {name}...")
        hist = train_one_schedule(name, factory, train_loader)
        results[name] = hist
        print(f"  Final Loss: {hist['loss'][-1]:.4f}, Final LR: {hist['lr'][-1]:.2e}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, EPOCHS + 1)

    # Plot 1: Learning Rate Schedules
    ax = axes[0]
    for name, hist in results.items():
        ax.plot(epochs_range, hist["lr"], label=name, linewidth=2)
    ax.set_yscale("log")
    ax.set_title("Learning Rate Schedules", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # Plot 2: Training Loss Curves
    ax = axes[1]
    for name, hist in results.items():
        ax.plot(epochs_range, hist["loss"], label=name, linewidth=2)
    ax.set_title("Training Loss Comparison", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Day 90: LR Schedule Impact on Optimization Dynamics", fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\nPlot saved to {PLOT_PATH}")