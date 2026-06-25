import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
OUTPUT_DIR = "day8_lr_schedules_output"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Simple Model & Data ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def get_dummy_data(n_samples=1000, input_dim=10):
    X = torch.randn(n_samples, input_dim)
    # Simple non-linear target: y = sin(sum(x)) + noise
    y = torch.sin(X.sum(dim=1, keepdim=True)) + 0.1 * torch.randn(n_samples, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Scheduler Definitions ---
def get_schedulers(optimizer):
    return {
        "Step Decay (step=15, gamma=0.1)": StepLR(optimizer, step_size=15, gamma=0.1),
        "Cosine Annealing (T_max=50)": CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6),
        "Cosine Warm Restarts (T_0=10, T_mult=2)": CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6),
    }

# --- Experiment Runner ---
def run_experiment(scheduler_name, scheduler, model, loader, criterion, optimizer, epochs):
    model.train()
    lr_history = []
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        loss_history.append(epoch_loss / len(loader))
        
    return lr_history, loss_history

# --- Visualization ---
def plot_results(results, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Day 8: Learning Rate Schedule Comparison", fontsize=16)
    
    # 1. LR Curves
    ax = axes[0, 0]
    for name, (lrs, _) in results.items():
        ax.plot(range(1, epochs + 1), lrs, label=name, linewidth=2)
    ax.set_yscale('log')
    ax.set_title("Learning Rate Schedules (Log Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Training Loss Curves
    ax = axes[0, 1]
    for name, (_, losses) in results.items():
        ax.plot(range(1, epochs + 1), losses, label=name, linewidth=2)
    ax.set_title("Training Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Zoomed LR (First 20 epochs)
    ax = axes[1, 0]
    for name, (lrs, _) in results.items():
        ax.plot(range(1, 21), lrs[:20], label=name, linewidth=2)
    ax.set_title("LR Detail: First 20 Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Loss Landscape Simulation (1D slice)
    # Visualize how different LRs traverse a simple quadratic bowl: f(x) = x^2
    ax = axes[1, 1]
    x_vals = np.linspace(-5, 5, 100)
    y_vals = x_vals ** 2
    ax.plot(x_vals, y_vals, 'k--', alpha=0.5, label="Loss Landscape $x^2$")
    
    # Simulate SGD steps for each schedule on f(x)=x^2, grad=2x
    x_start = 3.0
    for name, (lrs, _) in results.items():
        x = x_start
        trajectory = [x]
        for lr in lrs:
            grad = 2 * x
            x = x - lr * grad
            trajectory.append(x)
        ax.plot(trajectory, [t**2 for t in trajectory], 'o-', markersize=3, label=name, alpha=0.8)
    
    ax.set_title("Optimization Trajectory on $f(x)=x^2$ (Start x=3)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(OUTPUT_DIR, "lr_schedule_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()

# --- Main ---
if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    loader = get_dummy_data()
    criterion = nn.MSELoss()
    
    results = {}
    
    for name, scheduler_fn in get_schedulers(None).items():
        # Re-init model and optimizer for fair comparison
        model = SimpleMLP().to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        
        # Re-create scheduler with fresh optimizer
        if "Step" in name:
            scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        elif "Warm Restarts" in name:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
            
        print(f"\n--- Running: {name} ---")
        lrs, losses = run_experiment(name, scheduler, model, loader, criterion, optimizer, EPOCHS)
        results[name] = (lrs, losses)
        print(f"Final LR: {lrs[-1]:.6f}, Final Loss: {losses[-1]:.6f}")

    plot_results(results, EPOCHS)
    print("\nExperiment Complete.")