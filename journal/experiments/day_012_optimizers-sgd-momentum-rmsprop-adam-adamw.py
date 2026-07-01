import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_SAMPLES = 2000
NOISE = 0.15

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# Data Preparation
# ==========================================
X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train), torch.LongTensor(y_train)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_val), torch.LongTensor(y_val)
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# Model Definition
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
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

# ==========================================
# Optimizer Factory
# ==========================================
def get_optimizers(model):
    return {
        "SGD": optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        "SGD+Momentum": optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY),
        "RMSprop": optim.RMSprop(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        "Adam": optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        "AdamW": optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
    }

# ==========================================
# Training & Evaluation Loops
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

# ==========================================
# Main Experiment Runner
# ==========================================
def run_experiment():
    criterion = nn.CrossEntropyLoss()
    results = {}

    print(f"Running on {DEVICE} | Epochs: {EPOCHS} | LR: {LR}")
    print("-" * 60)

    for opt_name, opt_fn in get_optimizers(SimpleMLP().to(DEVICE)).items():
        # Re-init model for fair comparison
        model = SimpleMLP().to(DEVICE)
        optimizer = get_optimizers(model)[opt_name]

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)

            if epoch % 20 == 0 or epoch == 1:
                print(f"[{opt_name:14s}] Ep {epoch:3d}/{EPOCHS} | "
                      f"Tr Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        results[opt_name] = history
        print("-" * 60)

    return results

# ==========================================
# Plotting
# ==========================================
def plot_results(results, save_path="day12_optimizer_comparison.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Day 12: Optimizer Comparison on Moons Dataset", fontsize=16)

    metrics = [
        ("train_loss", "Training Loss", axes[0, 0]),
        ("val_loss", "Validation Loss", axes[0, 1]),
        ("train_acc", "Training Accuracy", axes[1, 0]),
        ("val_acc", "Validation Accuracy", axes[1, 1]),
    ]

    for key, title, ax in metrics:
        for name, hist in results.items():
            ax.plot(hist[key], label=name, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title.split()[-1])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if "loss" in key:
            ax.set_yscale("log")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()

# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)