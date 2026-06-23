import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 64
LR = 0.01
N_SAMPLES = 2000
NOISE = 0.25
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 1. Data Preparation ---
def get_data():
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    return train_loader, val_loader, X_val, y_val

# --- 2. Model Definition ---
class RegMLP(nn.Module):
    def __init__(self, use_dropout=False, use_batchnorm=False, dropout_p=0.5):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        
        self.fc1 = nn.Linear(2, 128)
        self.bn1 = nn.BatchNorm1d(128) if use_batchnorm else nn.Identity()
        self.drop1 = nn.Dropout(dropout_p) if use_dropout else nn.Identity()
        
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128) if use_batchnorm else nn.Identity()
        self.drop2 = nn.Dropout(dropout_p) if use_dropout else nn.Identity()
        
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        return x

    def l1_penalty(self):
        return sum(p.abs().sum() for p in self.parameters() if p.dim() > 1) # Only weights

# --- 3. Training & Evaluation ---
def train_one_epoch(model, loader, optimizer, criterion, l1_lambda=0.0):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        
        # Manual L1 Regularization
        if l1_lambda > 0:
            loss += l1_lambda * model.l1_penalty()
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total

def run_experiment(name, use_dropout, use_batchnorm, l2_lambda, l1_lambda):
    print(f"\n--- Running: {name} ---")
    train_loader, val_loader, _, _ = get_data()
    model = RegMLP(use_dropout=use_dropout, use_batchnorm=use_batchnorm).to(DEVICE)
    
    # L2 handled by optimizer weight_decay
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=l2_lambda)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, l1_lambda)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 50 == 0:
            gap = tr_acc - val_acc
            print(f"Ep {epoch+1:3d} | TrL:{tr_loss:.4f} ValL:{val_loss:.4f} | TrA:{tr_acc:.4f} ValA:{val_acc:.4f} | Gap:{gap:.4f}")
            
    return history, model

# --- 4. Decision Boundary Visualization ---
def plot_decision_boundary(model, X, y, ax, title):
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)
    
    with torch.no_grad():
        logits = model(grid)
        preds = logits.argmax(1).cpu().numpy().reshape(xx.shape)
    
    ax.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

# --- 5. Main Execution ---
if __name__ == "__main__":
    train_loader, val_loader, X_val, y_val = get_data()
    
    configs = [
        ("Baseline (No Reg)",       False, False, 0.0,    0.0),
        ("L2 Only (wd=1e-3)",       False, False, 1e-3,   0.0),
        ("L1 Only (lambda=1e-4)",   False, False, 0.0,    1e-4),
        ("Dropout (p=0.5)",         True,  False, 0.0,    0.0),
        ("BatchNorm Only",          False, True,  0.0,    0.0),
        ("Combo (BN+Drop+L2)",      True,  True,  1e-4,   0.0),
    ]
    
    all_histories = {}
    models = {}
    
    for name, do, bn, l2, l1 in configs:
        hist, model = run_experiment(name, do, bn, l2, l1)
        all_histories[name] = hist
        models[name] = model

    # --- Plotting Results ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    axes = axes.flatten()
    
    # 1. Loss Curves
    ax = axes[0]
    for name, hist in all_histories.items():
        ax.plot(hist['val_loss'], label=name, alpha=0.8)
    ax.set_title("Validation Loss Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.0)

    # 2. Accuracy Curves
    ax = axes[1]
    for name, hist in all_histories.items():
        ax.plot(hist['val_acc'], label=name, alpha=0.8)
    ax.set_title("Validation Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Generalization Gap (Train Acc - Val Acc) at End
    ax = axes[2]
    names = list(all_histories.keys())
    gaps = [all_histories[n]['train_acc'][-1] - all_histories[n]['val_acc'][-1] for n in names]
    bars = ax.barh(names, gaps, color='skyblue', edgecolor='navy')
    ax.set_title("Final Generalization Gap (Train Acc - Val Acc)")
    ax.set_xlabel("Gap (Lower is Better)")
    ax.invert_yaxis()
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{gap:.3f}", va='center')

    # 4. Weight Norms (L2 Norm of FC1 weights)
    ax = axes[3]
    norms = [models[n].fc1.weight.data.norm(2).item() for n in names]
    ax.barh(names, norms, color='lightcoral', edgecolor='darkred')
    ax.set_title("L2 Norm of First Layer Weights (||W||_2)")
    ax.set_xlabel("Weight Norm")
    ax.invert_yaxis()

    # 5. Decision Boundaries (Baseline vs Best Regularized)
    plot_decision_boundary(models["Baseline (No Reg)"], X_val, y_val, axes[4], "Baseline (Overfit)")
    plot_decision_boundary(models["Combo (BN+Drop+L2)"], X_val, y_val, axes[5], "Combo: BN + Dropout + L2")

    plt.tight_layout()
    plt.savefig("day7_regularization_results.png", dpi=150)
    print("\nPlot saved to 'day7_regularization_results.png'")
    plt.show()