import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SEED = 42
N_SAMPLES = 2000
INPUT_DIM = 2
HIDDEN_DIM = 64
OUTPUT_DIM = 2

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 1. Synthetic Dataset: Two Interleaving Half Circles (Moons) ---
def generate_moons(n_samples=2000, noise=0.1):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Outer circle
    theta_out = np.linspace(0, np.pi, n_samples_out)
    r_out = 1.0 + np.random.normal(0, noise, n_samples_out)
    x_out = np.stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)], axis=1)
    y_out = np.zeros(n_samples_out, dtype=np.int64)
    
    # Inner circle
    theta_in = np.linspace(0, np.pi, n_samples_in)
    r_in = 0.5 + np.random.normal(0, noise, n_samples_in)
    x_in = np.stack([r_in * np.cos(theta_in), -r_in * np.sin(theta_in) - 0.5], axis=1)
    y_in = np.ones(n_samples_in, dtype=np.int64)
    
    X = np.vstack([x_out, x_in])
    y = np.hstack([y_out, y_in])
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    return torch.FloatTensor(X[idx]), torch.LongTensor(y[idx])

X, y = generate_moons(N_SAMPLES, noise=0.15)
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
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

# --- 3. Training Function ---
def train_model(optimizer_class, optimizer_kwargs, name, epochs=EPOCHS):
    model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), lr=LR, **optimizer_kwargs)
    
    history = {'train_loss': [], 'time': []}
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * xb.size(0)
        
        avg_loss = epoch_loss / len(dataset)
        history['train_loss'].append(avg_loss)
        history['time'].append(time.time() - start_time)
        
    return history, model

# --- 4. Optimizer Configurations ---
optimizers_config = {
    "SGD":           (optim.SGD, {}),
    "SGD+Momentum":  (optim.SGD, {"momentum": 0.9}),
    "RMSProp":       (optim.RMSprop, {"alpha": 0.99}),
    "Adam":          (optim.Adam, {"betas": (0.9, 0.999), "eps": 1e-8}),
    "AdamW":         (optim.AdamW, {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-2}),
}

# --- 5. Run Experiments ---
results = {}
print(f"{'Optimizer':<18} | {'Final Loss':<12} | {'Time (s)':<10} | {'Loss @ Ep 10':<12} | {'Loss @ Ep 25':<12}")
print("-" * 75)

for name, (opt_class, kwargs) in optimizers_config.items():
    history, _ = train_model(opt_class, kwargs, name)
    results[name] = history
    
    final_loss = history['train_loss'][-1]
    total_time = history['time'][-1]
    loss_10 = history['train_loss'][9] if EPOCHS > 9 else 'N/A'
    loss_25 = history['train_loss'][24] if EPOCHS > 24 else 'N/A'
    
    print(f"{name:<18} | {final_loss:<12.6f} | {total_time:<10.2f} | {loss_10:<12.6f} | {loss_25:<12.6f}")

# --- 6. Plotting ---
plt.figure(figsize=(10, 6))
for name, history in results.items():
    plt.plot(history['train_loss'], label=name, linewidth=2)

plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Training Loss (Log Scale)")
plt.title("Day 94: Optimizer Convergence Comparison (Moons Dataset)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

plot_path = "optimizer_comparison_day94.png"
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {os.path.abspath(plot_path)}")

# --- 7. Decision Boundary Visualization (for best performer: usually Adam/AdamW) ---
# We pick AdamW for the boundary plot as it typically generalizes best with weight decay
print("\nGenerating decision boundary for AdamW...")
best_model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(best_model.parameters(), lr=LR, weight_decay=1e-2)
best_model.train()
for _ in range(EPOCHS):
    for xb, yb in dataloader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(best_model(xb), yb)
        loss.backward()
        optimizer.step()

best_model.eval()
# Create mesh grid
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)

with torch.no_grad():
    logits = best_model(grid)
    preds = torch.argmax(logits, dim=1).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.title("AdamW Decision Boundary (Day 94)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, alpha=0.3)

boundary_path = "decision_boundary_day94.png"
plt.savefig(boundary_path, dpi=150)
print(f"Decision boundary plot saved to {os.path.abspath(boundary_path)}")