import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict

# ==========================================
# Configuration & Reproducibility
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
NUM_SAMPLES = 5000
SAVE_DIR = "day53_optimizer_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 1. Synthetic Dataset: Noisy Spiral Classification
# ==========================================
def generate_spiral_data(n_samples=5000, noise=0.5):
    n = n_samples // 2
    # Class 0
    t = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    x1 = t * np.cos(t)
    y1 = t * np.sin(t)
    # Class 1
    t = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    x2 = -t * np.cos(t)
    y2 = -t * np.sin(t)
    
    X = np.vstack([np.stack([x1, y1]), np.stack([x2, y2])]).T
    y = np.hstack([np.zeros(n), np.ones(n)])
    
    # Shuffle
    idx = np.random.permutation(len(y))
    return torch.FloatTensor(X[idx]), torch.LongTensor(y[idx])

X, y = generate_spiral_data(NUM_SAMPLES)
dataset = torch.utils.data.TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. Model Definition
# ==========================================
class DeepMLP(nn.Module):
    """A deeper MLP to highlight optimizer differences in pathological curvature."""
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, depth=6):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) # Helps stability, but optimizers still behave differently
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Optimizer Factory
# ==========================================
def get_optimizers(model):
    return {
        "SGD": optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        "SGD_Momentum": optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY),
        "RMSProp": optim.RMSprop(model.parameters(), lr=LR, alpha=0.99, weight_decay=WEIGHT_DECAY, momentum=0.0),
        "Adam": optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY),
        "AdamW": optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY),
    }

# ==========================================
# 4. Training & Evaluation Loop
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

# ==========================================
# 5. Main Experiment Runner
# ==========================================
def run_experiment():
    criterion = nn.CrossEntropyLoss()
    history = defaultdict(lambda: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "time": []})
    
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | LR: {LR} | Weight Decay: {WEIGHT_DECAY}")
    print("-" * 60)

    for opt_name in ["SGD", "SGD_Momentum", "RMSProp", "Adam", "AdamW"]:
        print(f"\nTraining: {opt_name}...")
        model = DeepMLP().to(DEVICE)
        optimizers = get_optimizers(model)
        optimizer = optimizers[opt_name]
        
        start_time = time.time()
        best_val_acc = 0.0
        
        for epoch in range(1, EPOCHS + 1):
            tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            vl, va = evaluate(model, val_loader, criterion, DEVICE)
            
            history[opt_name]["train_loss"].append(tl)
            history[opt_name]["val_loss"].append(vl)
            history[opt_name]["train_acc"].append(ta)
            history[opt_name]["val_acc"].append(va)
            
            if va > best_val_acc:
                best_val_acc = va
            
            if epoch % 20 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(f"  Ep {epoch:3d} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f} | Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        history[opt_name]["time"].append(total_time)
        print(f"  >> Best Val Acc: {best_val_acc:.4f} | Total Time: {total_time:.1f}s")

    return history

# ==========================================
# 6. Plotting & Analysis
# ==========================================
def plot_results(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"SGD": "tab:gray", "SGD_Momentum": "tab:blue", "RMSProp": "tab:orange", "Adam": "tab:green", "AdamW": "tab:red"}
    linestyles = {"SGD": "--", "SGD_Momentum": "-", "RMSProp": "-.", "Adam": ":", "AdamW": "-"}
    
    epochs_range = range(1, EPOCHS + 1)
    
    # 1. Training Loss (Log Scale)
    ax = axes[0, 0]
    for name, hist in history.items():
        ax.plot(epochs_range, hist["train_loss"], label=name, color=colors[name], linestyle=linestyles[name], alpha=0.8)
    ax.set_yscale('log')
    ax.set_title("Training Loss (Log Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # 2. Validation Loss
    ax = axes[0, 1]
    for name, hist in history.items():
        ax.plot(epochs_range, hist["val_loss"], label=name, color=colors[name], linestyle=linestyles[name], alpha=0.8)
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Validation Accuracy
    ax = axes[1, 0]
    for name, hist in history.items():
        ax.plot(epochs_range, hist["val_acc"], label=name, color=colors[name], linestyle=linestyles[name], alpha=0.8)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Final Performance Bar Chart
    ax = axes[1, 1]
    names = list(history.keys())
    final_accs = [history[n]["val_acc"][-1] for n in names]
    best_accs = [max(history[n]["val_acc"]) for n in names]
    times = [history[n]["time"][0] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, final_accs, width, label='Final Acc', color='skyblue', edgecolor='black')
    ax.bar(x + width/2, best_accs, width, label='Best Acc', color='salmon', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Final vs Best Validation Accuracy")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Annotate times
    for i, t in enumerate(times):
        ax.text(i, max(final_accs[i], best_accs[i]) + 0.01, f"{t:.0f}s", ha='center', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "optimizer_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()

def print_summary_table(history):
    print("\n" + "="*80)
    print(f"{'Optimizer':<15} | {'Final Val Acc':>14} | {'Best Val Acc':>13} | {'Final Train Acc':>16} | {'Time (s)':>8}")
    print("-"*80)
    for name, hist in history.items():
        print(f"{name:<15} | {hist['val_acc'][-1]:>14.4f} | {max(hist['val_acc']):>13.4f} | {hist['train_acc'][-1]:>16.4f} | {hist['time'][0]:>8.1f}")
    print("="*80)

# ==========================================
# 7. Decision Boundary Visualization (Qualitative)
# ==========================================
def plot_decision_boundaries(history):
    # Retrain best models for visualization (or just use last state)
    # For simplicity, we retrain quickly the best performer (usually AdamW) and worst (SGD) for visual contrast
    # Actually, let's plot all 5 in a grid.
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    for idx, opt_name in enumerate(["SGD", "SGD_Momentum", "RMSProp", "Adam", "AdamW"]):
        model = DeepMLP().to(DEVICE)
        optimizer = get_optimizers(model)[opt_name]
        
        # Quick train (20 epochs) just for viz, or load full history state? 
        # Retraining 20 epochs is fast and shows "early" behavior.
        for _ in range(30): 
            train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            
        model.eval()
        with torch.no_grad():
            Z = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)
        
        ax = axes[idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu, levels=[-0.5, 0.5, 1.5])
        ax.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap=plt.cm.RdYlBu, edgecolors='k', linewidth=0.2, alpha=0.6)
        ax.set_title(opt_name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    plt.suptitle("Decision Boundaries after 30 Epochs (Qualitative)", fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "decision_boundaries.png")
    plt.savefig(save_path, dpi=150)
    print(f"Decision boundary plot saved to {save_path}")
    plt.close()

# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("DAY 53: OPTIMIZER COMPARISON MINI-EXPERIMENT")
    print("="*60)
    
    hist = run_experiment()
    plot_results(hist)
    print_summary_table(hist)
    plot_decision_boundaries(hist)
    
    print("\nExperiment Complete.")