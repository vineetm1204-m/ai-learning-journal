import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

def target_function(x):
    return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x) + 0.3 * x

class ShallowNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(1, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))

def train_model(hidden_size, epochs=2000, lr=0.01):
    model = ShallowNet(hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_train = torch.linspace(-1, 1, 200).unsqueeze(1)
    y_train = torch.tensor(target_function(x_train.numpy()), dtype=torch.float32)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return model, losses

def evaluate_model(model):
    x_test = torch.linspace(-1.5, 1.5, 500).unsqueeze(1)
    with torch.no_grad():
        y_pred = model(x_test).numpy().flatten()
    y_true = target_function(x_test.numpy())
    mse = np.mean((y_pred - y_true) ** 2)
    return x_test.numpy().flatten(), y_pred, y_true, mse

hidden_sizes = [2, 5, 10, 20, 50]
results = {}

print("Universal Approximation Theorem - Mini Experiment")
print("=" * 55)
print(f"{'Hidden Units':<15} {'Final Loss':<15} {'Test MSE':<15}")
print("-" * 55)

for h in hidden_sizes:
    model, losses = train_model(h, epochs=3000, lr=0.02)
    x_vals, y_pred, y_true, mse = evaluate_model(model)
    results[h] = (x_vals, y_pred, y_true, mse)
    print(f"{h:<15} {losses[-1]:<15.6f} {mse:<15.6f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

x_dense = np.linspace(-1.5, 1.5, 500)
y_dense = target_function(x_dense)

for idx, h in enumerate(hidden_sizes):
    ax = axes[idx]
    x_vals, y_pred, y_true, mse = results[h]
    ax.plot(x_dense, y_dense, 'k--', label='True function', alpha=0.7, linewidth=2)
    ax.plot(x_vals, y_pred, 'r-', label=f'NN (h={h})', linewidth=1.5)
    ax.set_title(f'Hidden Units: {h} | MSE: {mse:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].axis('off')
axes[-1].text(0.1, 0.5, 
    "Universal Approximation Theorem:\n\n"
    "A feedforward network with a single hidden layer\n"
    "and finite neurons can approximate any continuous\n"
    "function on compact subsets of ℝⁿ, given:\n\n"
    "• Non-polynomial activation (e.g., tanh, ReLU)\n"
    "• Sufficient hidden units\n\n"
    "This experiment demonstrates the theorem empirically:\n"
    "as hidden units increase, approximation error decreases.",
    fontsize=11, verticalalignment='center', fontfamily='monospace')

plt.suptitle('Day 10: Universal Approximation Theorem', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('day10_universal_approximation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nExperiment complete. Plot saved as 'day10_universal_approximation.png'")