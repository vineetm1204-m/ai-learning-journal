import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Synthetic Dataset Generation (Noisy Spiral / Moons style for clear overfitting visualization)
def generate_data(n_samples=2000, noise=0.3, test_split=0.2):
    # Two interleaving half circles (make_moons equivalent)
    n = n_samples // 2
    t = np.linspace(0, 2 * np.pi, n)
    x1 = np.stack([np.cos(t), np.sin(t)], axis=1)
    x2 = np.stack([-np.cos(t) + 1, -np.sin(t) - 0.5], axis=1)
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n), np.ones(n)])
    
    # Add noise
    X += np.random.randn(*X.shape) * noise
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    split = int(n_samples * (1 - test_split))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return (torch.FloatTensor(X_train), torch.LongTensor(y_train),
            torch.FloatTensor(X_test), torch.LongTensor(y_test))

X_train, y_train, X_test, y_test = generate_data(2000, noise=0.4)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

# --- Model Definitions ---
class BaseMLP(nn.Module):
    """A reasonably deep MLP prone to overfitting on small noisy data."""
    def __init__(self, input_dim=2, hidden_dims=[128, 128, 64, 32], output_dim=2, 
                 use_bn=False, dropout_p=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_weights(self):
        """Return all weight matrices (for L1/L2 calc)."""
        return [p for n, p in self.named_parameters() if 'weight' in n and 'bn' not in n]

# --- Regularized Loss Function ---
def compute_loss(model, outputs, targets, criterion, l1_lambda, l2_lambda):
    loss = criterion(outputs, targets)
    if l1_lambda > 0:
        l1_norm = sum(p.abs().sum() for p in model.get_weights())
        loss += l1_lambda * l1_norm
    if l2_lambda > 0:
        l2_norm = sum(p.pow(2).sum() for p in model.get_weights())
        loss += l2_lambda * l2_norm
    return loss

# --- Training Loop ---
def train_model(model, train_loader, test_loader, config, epochs=50, lr=1e-3):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.get('wd', 0)) # weight_decay is native L2
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    l1_lambda = config.get('l1', 0)
    l2_lambda = config.get('l2', 0) # Manual L2 (if not using optimizer weight_decay)
    
    # Note: If using optimizer weight_decay, don't double count manual L2.
    # Here we use manual L2 via config['l2'] and keep optimizer wd=0 for clarity.

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = compute_loss(model, out, yb, criterion, l1_lambda, l2_lambda)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        
        history['train_loss'].append(train_loss / total)
        history['train_acc'].append(correct / total)
        
        # Validation
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                # Eval loss usually reported without reg terms, but we include for consistency
                loss = compute_loss(model, out, yb, criterion, l1_lambda, l2_lambda)
                test_loss += loss.item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total += xb.size(0)
        
        history['test_loss'].append(test_loss / total)
        history['test_acc'].append(correct / total)
        
    return history

# --- Experiment Configs ---
configs = {
    "Baseline (No Reg)":      {'l1': 0, 'l2': 0, 'dropout': 0.0, 'bn': False, 'wd': 0},
    "L1 Regularization":      {'l1': 1e-4, 'l2': 0, 'dropout': 0.0, 'bn': False, 'wd': 0},
    "L2 Regularization":      {'l1': 0, 'l2': 1e-4, 'dropout': 0.0, 'bn': False, 'wd': 0},
    "Dropout (p=0.3)":        {'l1': 0, 'l2': 0, 'dropout': 0.3, 'bn': False, 'wd': 0},
    "BatchNorm":              {'l1': 0, 'l2': 0, 'dropout': 0.0, 'bn': True, 'wd': 0},
    "Combo (L2+Dropout+BN)":  {'l1': 0, 'l2': 1e-5, 'dropout': 0.2, 'bn': True, 'wd': 0},
}

# --- Run Experiments ---
results = {}
print(f"Running on {DEVICE}...")
for name, cfg in configs.items():
    print(f"\n--- Training: {name} ---")
    model = BaseMLP(dropout_p=cfg['dropout'], use_bn=cfg['bn'])
    hist = train_model(model, train_loader, test_loader, cfg, epochs=60, lr=1e-3)
    results[name] = hist
    print(f"Final Test Acc: {hist['test_acc'][-1]:.4f} | Gap (Train-Test Acc): {hist['train_acc'][-1] - hist['test_acc'][-1]:.4f}")

# --- Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Day 89: Regularization Comparison (Synthetic Noisy Moons)', fontsize=16)

metrics = [
    ('train_loss', 'Test Loss', axes[0, 0]),
    ('train_acc', 'Train Accuracy', axes[0, 1]),
    ('test_loss', 'Test Loss', axes[1, 0]),
    ('test_acc', 'Test Accuracy', axes[1, 1]),
]

# Note: We plot train_loss and test_loss on separate axes for clarity, 
# but usually train_loss is lower. Let's plot Train vs Test Loss on one, Train vs Test Acc on other.
# Adjusting layout:
# axes[0,0]: Train Loss vs Test Loss
# axes[0,1]: Train Acc vs Test Acc
# axes[1,0]: Generalization Gap (Loss)
# axes[1,1]: Generalization Gap (Acc)

colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

for i, (name, hist) => enumerate(configs.items()):
    c = colors[i]
    epochs_range = range(1, len(hist['train_loss']) + 1)
    
    # Loss Curves
    axes[0, 0].plot(epochs_range, hist['train_loss'], label=f'{name} (Train)', color=c, linestyle='--', alpha=0.7)
    axes[0, 0].plot(epochs_range, hist['test_loss'], label=f'{name} (Test)', color=c, alpha=0.9)
    
    # Acc Curves
    axes[0, 1].plot(epochs_range, hist['train_acc'], label=f'{name} (Train)', color=c, linestyle='--', alpha=0.7)
    axes[0, 1].plot(epochs_range, hist['test_acc'], label=f'{name} (Test)', color=c, alpha=0.9)
    
    # Generalization Gap
    gap_loss = np.array(hist['test_loss']) - np.array(hist['train_loss'])
    gap_acc = np.array(hist['train_acc']) - np.array(hist['test_acc'])
    axes[1, 0].plot(epochs_range, gap_loss, label=name, color=c)
    axes[1, 1].plot(epochs_range, gap_acc, label=name, color=c)

axes[0, 0].set_title('Loss Curves (Solid=Test, Dashed=Train)')
axes[0, 0].set_ylabel('Cross Entropy Loss')
axes[0, 0].set_yscale('log')
axes[0, 0].legend(fontsize=7, ncol=2)

axes[0, 1].set_title('Accuracy Curves (Solid=Test, Dashed=Train)')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_ylim(0, 1.05)

axes[1, 0].set_title('Generalization Gap (Test Loss - Train Loss)')
axes[1, 0].set_ylabel('Gap (Lower is Better)')
axes[1, 0].axhline(0, color='k', linestyle=':', alpha=0.3)

axes[1, 1].set_title('Generalization Gap (Train Acc - Test Acc)')
axes[1, 1].set_ylabel('Gap (Lower is Better)')
axes[1, 1].axhline(0, color='k', linestyle=':', alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('day89_regularization_results.png', dpi=150)
print("\nPlot saved to 'day89_regularization_results.png'")

# --- Weight Sparsity Analysis (L1 Specific) ---
print("\n--- Weight Sparsity Analysis (L1 vs Others) ---")
l1_model = BaseMLP()
_ = train_model(l1_model, train_loader, test_loader, configs["L1 Regularization"], epochs=10, lr=1e-3) # Quick train
baseline_model = BaseMLP()
_ = train_model(baseline_model, train_loader, test_loader, configs["Baseline (No Reg)"], epochs=10, lr=1e-3)

def count_near_zero(model, threshold=1e-4):
    total, near_zero = 0, 0
    for p in model.get_weights():
        total += p.numel()
        near_zero += (p.abs() < threshold).sum().item()
    return near_zero / total * 100

print(f"Baseline % weights < 1e-4: {count_near_zero(baseline_model):.2f}%")
print(f"L1 Model % weights < 1e-4: {count_near_zero(l1_model):.2f}%")

# --- Decision Boundary Visualization (Best Model) ---
best_name = max(results, key=lambda k: results[k]['test_acc'][-1])
print(f"\nBest Config: {best_name} (Test Acc: {results[best_name]['test_acc'][-1]:.4f})")

best_cfg = configs[best_name]
best_model = BaseMLP(dropout_p=best_cfg['dropout'], use_bn=best_cfg['bn'])
_ = train_model(best_model, train_loader, test_loader, best_cfg, epochs=60, lr=1e-3)

# Plot Decision Boundary
h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)

best_model.eval()
with torch.no_grad():
    Z = best_model(grid).argmax(1).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', cmap=plt.cm.coolwarm, label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, marker='^', edgecolor='k', cmap=plt.cm.coolwarm, alpha=0.6, label='Test')
plt.title(f'Decision Boundary: {best_name}')
plt.legend()
plt.savefig('day89_decision_boundary.png', dpi=150)
print("Decision boundary saved to 'day89_decision_boundary.png'")
plt.show()