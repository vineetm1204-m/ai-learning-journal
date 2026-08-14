import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Configuration & Data Preparation
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 2000
NOISE = 0.25
TEST_SIZE = 0.3
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.01
HIDDEN_DIM = 128
N_LAYERS = 4  # Deep enough to overfit easily

print(f"Using device: {DEVICE}")

# Generate synthetic non-linear data (Two Moons)
X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# Convert to Tensor Datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_train).to(DEVICE), torch.LongTensor(y_train).to(DEVICE)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_val).to(DEVICE), torch.LongTensor(y_val).to(DEVICE)
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. Model Definition
# ==========================================
class RegMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, n_layers=4, 
                 use_dropout=False, dropout_p=0.5, use_batchnorm=False):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_p))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def l1_penalty(self):
        return sum(p.abs().sum() for p in self.parameters() if p.dim() > 1) # Only weights

# ==========================================
# 3. Training & Evaluation Logic
# ==========================================
def train_epoch(model, loader, optimizer, criterion, l1_lambda=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Manual L1 Regularization
        if l1_lambda > 0:
            loss += l1_lambda * model.l1_penalty()
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += X_batch.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += X_batch.size(0)
    return total_loss / total, correct / total

def run_experiment(config_name, model_kwargs, optimizer_kwargs, l1_lambda=0.0):
    print(f"\n--- Running: {config_name} ---")
    model = RegMLP(**model_kwargs).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, l1_lambda)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            gap = tr_acc - val_acc
            print(f"Ep {epoch:02d} | TrL:{tr_loss:.4f} ValL:{val_loss:.4f} | TrA:{tr_acc:.4f} ValA:{val_acc:.4f} | Gap:{gap:.4f}")
            
    return model, history

# ==========================================
# 4. Experiment Configurations
# ==========================================
base_model_args = {'hidden_dim': HIDDEN_DIM, 'n_layers': N_LAYERS}
base_opt_args = {'lr': LR, 'weight_decay': 0.0} # L2 handled here

configs = {
    "Baseline (No Reg)":       ({**base_model_args, 'use_dropout': False, 'use_batchnorm': False}, {**base_opt_args}, 0.0),
    "L2 (Weight Decay=1e-3)":  ({**base_model_args, 'use_dropout': False, 'use_batchnorm': False}, {**base_opt_args, 'weight_decay': 1e-3}, 0.0),
    "L1 (Lambda=1e-4)":        ({**base_model_args, 'use_dropout': False, 'use_batchnorm': False}, {**base_opt_args}, 1e-4),
    "Dropout (p=0.3)":         ({**base_model_args, 'use_dropout': True, 'dropout_p': 0.3, 'use_batchnorm': False}, {**base_opt_args}, 0.0),
    "BatchNorm":               ({**base_model_args, 'use_dropout': False, 'use_batchnorm': True}, {**base_opt_args}, 0.0),
    "Combo (BN + DO + L2)":    ({**base_model_args, 'use_dropout': True, 'dropout_p': 0.2, 'use_batchnorm': True}, {**base_opt_args, 'weight_decay': 1e-4}, 0.0),
}

results = {}
for name, (m_args, o_args, l1) in configs.items():
    model, hist = run_experiment(name, m_args, o_args, l1)
    results[name] = hist

# ==========================================
# 5. Visualization
# ==========================================
def plot_decision_boundary(model, X, y, title, ax):
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)
    
    with torch.no_grad():
        logits = model(grid)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    preds = preds.reshape(xx.shape)
    ax.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

# Plot 1-6: Loss Curves
for idx, (name, hist) in enumerate(results.items()):
    ax = axes[idx]
    epochs_range = range(1, EPOCHS + 1)
    ax.plot(epochs_range, hist['train_loss'], label='Train Loss', color='blue', alpha=0.7)
    ax.plot(epochs_range, hist['val_loss'], label='Val Loss', color='orange', alpha=0.7)
    ax.set_title(f"{name}\n(Final Val Acc: {hist['val_acc'][-1]:.3f})", fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Plot 7-12: Decision Boundaries (using last trained model from loop, need to retrain or store models)
# For simplicity, we retrain quickly for visualization on the last config or just plot the last one.
# Better: Store models in results dict. Let's modify loop slightly to store model.
# Re-running last config for boundary plots for all configs would be slow.
# Instead, let's plot boundaries for a subset: Baseline, Dropout, BatchNorm, Combo.

print("\nGenerating Decision Boundary Plots...")
boundary_configs = ["Baseline (No Reg)", "Dropout (p=0.3)", "BatchNorm", "Combo (BN + DO + L2)"]
for i, name in enumerate(boundary_configs):
    # Retrain quickly for boundary (or retrieve if stored)
    # We'll just train a fresh one for 30 epochs for viz to keep code self-contained without storing heavy models
    m_args, o_args, l1 = configs[name]
    viz_model, _ = run_experiment(f"Viz-{name}", m_args, o_args, l1) # Reuses function but prints less if we modify... 
    # To avoid spam, let's just do a silent train here.
    viz_model = RegMLP(**m_args).to(DEVICE)
    opt = optim.Adam(viz_model.parameters(), **o_args)
    crit = nn.CrossEntropyLoss()
    for _ in range(30): train_epoch(viz_model, train_loader, opt, crit, l1)
    
    plot_decision_boundary(viz_model, X_val, y_val, name, axes[6 + i])
    axes[6 + i].set_title(f"Boundary: {name}")

# Hide unused subplots
for j in range(10, 12):
    axes[j].axis('off')

plt.suptitle("Day 48: Regularization Comparison (L1, L2, Dropout, BatchNorm)", fontsize=16, y=1.0)
plt.tight_layout()
plt.show()

# ==========================================
# 6. Summary Table
# ==========================================
print("\n" + "="*60)
print(f"{'Configuration':<25} | {'Train Acc':>10} | {'Val Acc':>10} | {'Gap':>8}")
print("-"*60)
for name, hist in results.items():
    tr = hist['train_acc'][-1]
    vl = hist['val_acc'][-1]
    print(f"{name:<25} | {tr:>10.4f} | {vl:>10.4f} | {tr-vl:>8.4f}")
print("="*60)