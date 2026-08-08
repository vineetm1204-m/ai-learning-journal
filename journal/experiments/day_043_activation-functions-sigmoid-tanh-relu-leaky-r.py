import numpy as np
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def gelu(x):
    # GELU approximation using tanh (as in original paper)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    # Derivative of the approximate GELU
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    coef = 0.044715
    x3 = x ** 3
    tanh_arg = sqrt_2_over_pi * (x + coef * x3)
    tanh_val = np.tanh(tanh_arg)
    sech2 = 1 - tanh_val ** 2
    derivative = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * sqrt_2_over_pi * (1 + 3 * coef * x ** 2)
    return derivative

# Plotting
def plot_activations():
    x = np.linspace(-5, 5, 1000)
    functions = [
        ("Sigmoid", sigmoid, sigmoid_derivative),
        ("Tanh", tanh, tanh_derivative),
        ("ReLU", relu, relu_derivative),
        ("Leaky ReLU (α=0.01)", leaky_relu, leaky_relu_derivative),
        ("GELU", gelu, gelu_derivative),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Activation Functions and Their Derivatives", fontsize=16)

    for idx, (name, fn, deriv) in enumerate(functions):
        y = fn(x)
        dy = deriv(x)

        # Activation
        ax = axes[0, idx]
        ax.plot(x, y, label=name, color='tab:blue')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5) if name != "ReLU" and name != "Leaky ReLU (α=0.01)" else ax.set_ylim(-0.5, 5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        # Derivative
        ax = axes[1, idx]
        ax.plot(x, dy, label=f"d{name}/dx", color='tab:orange')
        ax.set_title(f"Derivative of {name}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 1.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig("activation_functions_day43.png", dpi=150)
    print("Saved plot to activation_functions_day43.png")
    plt.close()

# Gradient flow simulation through a deep linear network with activations
def simulate_gradient_flow(num_layers=10, hidden_dim=128, batch_size=64, seed=42):
    np.random.seed(seed)
    x = np.random.randn(batch_size, hidden_dim)
    # Initialize weights with Xavier/Glorot for fair comparison
    weights = []
    for _ in range(num_layers):
        w = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        weights.append(w)

    activations = [
        ("Sigmoid", sigmoid, sigmoid_derivative),
        ("Tanh", tanh, tanh_derivative),
        ("ReLU", relu, relu_derivative),
        ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
        ("GELU", gelu, gelu_derivative),
    ]

    print("\n=== Gradient Flow Simulation (Mean Gradient Norm per Layer) ===")
    print(f"Network: {num_layers} layers, hidden_dim={hidden_dim}, batch_size={batch_size}")
    print("-" * 70)

    for name, fn, deriv in activations:
        # Forward pass
        h = x.copy()
        layer_outputs = [h]
        for w in weights:
            h = h @ w
            h = fn(h)
            layer_outputs.append(h)

        # Backward pass: assume loss gradient = 1 at output
        grad = np.ones_like(h)
        grad_norms = []
        for i in reversed(range(num_layers)):
            # Gradient through activation
            grad = grad * deriv(layer_outputs[i + 1])
            # Gradient through weight matrix
            grad = grad @ weights[i].T
            grad_norms.append(np.mean(np.linalg.norm(grad, axis=1)))

        grad_norms = grad_norms[::-1]  # from input to output
        print(f"{name:15s}: " + " | ".join(f"{g:.4f}" for g in grad_norms))

# Simple classification experiment on a synthetic dataset
def classification_experiment():
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Generate data
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simple 2-layer MLP using numpy (no autograd)
    class SimpleMLP:
        def __init__(self, input_dim, hidden_dim, output_dim, activation_fn, activation_deriv, lr=0.01):
            self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
            self.b2 = np.zeros(output_dim)
            self.activation = activation_fn
            self.activation_deriv = activation_deriv
            self.lr = lr

        def forward(self, X):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = self.activation(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
            # Softmax
            exp_z2 = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
            self.probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
            return self.probs

        def backward(self, X, y):
            m = X.shape[0]
            # Output layer gradient
            dz2 = self.probs.copy()
            dz2[np.arange(m), y] -= 1
            dz2 /= m
            dW2 = self.a1.T @ dz2
            db2 = np.sum(dz2, axis=0)

            # Hidden layer gradient
            da1 = dz2 @ self.W2.T
            dz1 = da1 * self.activation_deriv(self.z1)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)

            # Update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            # Loss
            loss = -np.mean(np.log(self.probs[np.arange(m), y] + 1e-8))
            return loss

        def predict(self, X):
            probs = self.forward(X)
            return np.argmax(probs, axis=1)

    activations = [
        ("Sigmoid", sigmoid, sigmoid_derivative),
        ("Tanh", tanh, tanh_derivative),
        ("ReLU", relu, relu_derivative),
        ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
        ("GELU", gelu, gelu_derivative),
    ]

    print("\n=== Classification Experiment on Moons Dataset ===")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("-" * 70)

    for name, fn, deriv in activations:
        model = SimpleMLP(input_dim=2, hidden_dim=64, output_dim=2, activation_fn=fn, activation_deriv=deriv, lr=0.05)
        losses = []
        for epoch in range(200):
            loss = model.backward(X_train, y_train)
            losses.append(loss)
        train_acc = np.mean(model.predict(X_train) == y_train)
        test_acc = np.mean(model.predict(X_test) == y_test)
        print(f"{name:15s}: Final Loss={losses[-1]:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

if __name__ == "__main__":
    print("Day 43: Activation Functions Mini-Experiment")
    print("=" * 50)

    # 1. Plot activations and derivatives
    plot_activations()

    # 2. Gradient flow simulation
    simulate_gradient_flow()

    # 3. Classification experiment
    classification_experiment()

    print("\nExperiment completed.")