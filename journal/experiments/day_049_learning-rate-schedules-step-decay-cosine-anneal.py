import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
OUTPUT_DIR = "lr_schedule_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 1. Synthetic Dataset ---
def get_data_loaders(n_samples=2000, input_dim=20, n_classes=5):
    X = torch.randn(n_samples, input_dim)
    # Create a non-trivial decision boundary
    y = (X[:, 0] * 2 + X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 3] + torch.randn(n_samples) * 0.5 > 0).long()
    # Make it multi-class for more interesting loss landscape
    y = (y * 2 + (X[:, 4] > 0).long()) % n_classes 
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

# --- 2. Simple Model ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# --- 3. Training Loop per Scheduler ---
def run_experiment(scheduler_name, scheduler_fn, optimizer_fn, train_loader, epochs=EPOCHS):
    model = SimpleMLP().to(DEVICE)
    optimizer = optimizer_fn(model.parameters())
    scheduler = scheduler_fn(optimizer)
    criterion = nn.CrossEntropyLoss()
    
    history = {"lr": [], "loss": []}
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * xb.size(0)
        
        # Step scheduler (per epoch for these schedulers)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / len(train_loader.dataset)
        
        history["lr"].append(current_lr)
        history["loss"].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  [{scheduler_name}] Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f}")
            
    return history

# --- 4. Scheduler Definitions ---
def get_schedulers(optimizer):
    return {
        "Step Decay (gamma=0.1, step=15)": 
            lambda opt: StepLR(opt, step_size=15, gamma=0.1),
        "Cosine Annealing (T_max=50)": 
            lambda opt: CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6),
        "Cosine Warm Restarts (T_0=10, T_mult=2)": 
            lambda opt: CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6),
    }

# --- 5. Main Execution ---
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    train_loader = get_data_loaders()
    optimizer_fn = lambda params: optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    schedulers = get_schedulers(None) # Pass None, we bind optimizer inside loop
    
    results = {}
    
    for name, scheduler_factory in schedulers.items():
        print(f"\n--- Running: {name} ---")
        # We need a fresh optimizer factory for each run to reset state
        hist = run_experiment(name, scheduler_factory, optimizer_fn, train_loader)
        results[name] = hist
    
    # --- 6. Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Day 49: Learning Rate Schedule Comparison", fontsize=16)
    
    # Plot 1: LR Schedules
    ax = axes[0, 0]
    for name, hist in results.items():
        ax.plot(hist["lr"], label=name, linewidth=2)
    ax.set_yscale("log")
    ax.set_title("Learning Rate Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    ax = axes[0, 1]
    for name, hist in results.items():
        ax.plot(hist["loss"], label=name, linewidth=2)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss (Log Scale)
    ax = axes[1, 0]
    for name, hist in results.items():
        ax.plot(hist["loss"], label=name, linewidth=2)
    ax.set_yscale("log")
    ax.set_title("Training Loss (Log Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: LR vs Loss Scatter (Last 20 epochs)
    ax = axes[1, 1]
    for name, hist in results.items():
        lrs = hist["lr"][-20:]
        losses = hist["loss"][-20:]
        ax.scatter(lrs, losses, label=name, alpha=0.7, s=30)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("LR vs Loss (Last 20 Epochs)")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(OUTPUT_DIR, "lr_schedule_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    
    # Print final stats
    print("\n--- Final Statistics (Last Epoch) ---")
    for name, hist in results.items():
        print(f"{name:40s} | Final LR: {hist['lr'][-1]:.2e} | Final Loss: {hist['loss'][-1]:.4f}")