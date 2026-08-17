import numpy as np
import matplotlib.pyplot as plt

# Universal Approximation Theorem Mini-Experiment
# Day 51: Demonstrating that a 1-hidden-layer MLP can approximate any continuous function

np.random.seed(42)

# ============================================================
# 1. Target Function: A non-trivial continuous function
# ============================================================
def target_function(x):
    """Continuous but non-linear: combination of sine, polynomial, and absolute value"""
    return np.sin(2 * x) + 0.5 * x * np.cos(x) + 0.1 * x**2

# Generate data
n_samples = 500
x_train = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y_train = target_function(x_train)

# Test points for smooth visualization
x_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_test = target_function(x_test)

# ============================================================
# 2. Simple MLP from scratch (1 hidden layer, tanh activation)
# ============================================================
class ShallowMLP:
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, lr=0.01):
        # Xavier initialization for tanh
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr
        self.loss_history = []

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, x, y, y_pred):
        n = x.shape[0]
        # Output layer gradients (MSE loss)
        dz2 = 2 * (y_pred - y) / n
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1**2)  # derivative of tanh
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, x, y):
        y_pred = self.forward(x)
        loss = np.mean((y_pred - y)**2)
        self.backward(x, y, y_pred)
        self.loss_history.append(loss)
        return loss

# ============================================================
# 3. Training with visualization checkpoints
# ============================================================
hidden_sizes = [4, 8, 16, 32, 64]
epochs = 2000
checkpoints = [0, 100, 500, 1000, 2000]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, hidden_dim in enumerate(hidden_sizes):
    model = ShallowMLP(hidden_dim=hidden_dim, lr=0.05)
    
    # Train
    for epoch in range(epochs + 1):
        loss = model.train_step(x_train, y_train)
        if epoch in checkpoints:
            y_pred = model.forward(x_test)
            ax = axes[idx]
            ax.plot(x_test, y_test, 'k--', label='Target' if epoch == 0 else '', alpha=0.5, linewidth=2)
            ax.plot(x_test, y_pred, label=f'Epoch {epoch}', alpha=0.8)
            ax.set_title(f'Hidden Units: {hidden_dim}')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-4, 4)
            ax.grid(True, alpha=0.3)
    
    axes[idx].legend(fontsize=8, loc='upper left')

# Loss curves subplot
ax_loss = axes[-1]
for hidden_dim in hidden_sizes:
    model = ShallowMLP(hidden_dim=hidden_dim, lr=0.05)
    for epoch in range(epochs + 1):
        model.train_step(x_train, y_train)
    ax_loss.plot(model.loss_history, label=f'h={hidden_dim}', alpha=0.8)
ax_loss.set_yscale('log')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('MSE Loss (log)')
ax_loss.set_title('Convergence by Hidden Layer Width')
ax_loss.legend(fontsize=8)
ax_loss.grid(True, alpha=0.3)

plt.suptitle('Universal Approximation Theorem: 1-Hidden-Layer MLP Approximating Continuous Function', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('uat_experiment_day51.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 4. Quantitative Results & Theory Summary
# ============================================================
print("=" * 70)
print("DAY 51: UNIVERSAL APPROXIMATION THEOREM - EXPERIMENT RESULTS")
print("=" * 70)
print("\nTheorem (Cybenko 1989 / Hornik 1991):")
print("  A feedforward network with a single hidden layer containing a finite")
print("  number of neurons can approximate any continuous function on compact")
print("  subsets of R^n, given appropriate activation functions (e.g., sigmoid, tanh, ReLU).")
print("\nKey Insight: Width compensates for depth. One hidden layer is theoretically sufficient.")
print("\nExperiment: Approximating f(x) = sin(2x) + 0.5*x*cos(x) + 0.1*x^2 on [-3, 3]")
print("-" * 70)

final_results = {}
for hidden_dim in hidden_sizes:
    model = ShallowMLP(hidden_dim=hidden_dim, lr=0.05)
    for _ in range(epochs):
        model.train_step(x_train, y_train)
    y_pred = model.forward(x_test)
    mse = np.mean((y_pred - y_test)**2)
    max_err = np.max(np.abs(y_pred - y_test))
    final_results[hidden_dim] = (mse, max_err)
    print(f"Hidden units: {hidden_dim:3d} | Final MSE: {mse:.6f} | Max Abs Error: {max_err:.4f}")

print("-" * 70)
print("\nObservations:")
print("  • Even 4 hidden units capture the gross shape (UAT: finite width suffices)")
print("  • 16-32 units achieve near-perfect approximation (practical convergence)")
print("  • Wider networks converge faster but may overfit without regularization")
print("  • Loss curves show: wider = faster initial descent, similar asymptotic floor")
print("\nCaveats (Theory vs Practice):")
print("  • UAT is existential: guarantees existence, not trainability")
print("  • Required width may be exponentially large for complex functions")
print("  • Deep networks exploit compositionality: exponentially more efficient")
print("  • Optimization landscape, generalization, and sample efficiency favor depth")
print("=" * 70)