import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Activation functions and derivatives
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

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'relu': (relu, relu_derivative)
}

# Weight initializations
def init_weights(shape, method='standard'):
    fan_in, fan_out = shape[1], shape[0]
    if method == 'standard':
        return np.random.randn(*shape) * 0.01
    elif method == 'xavier':
        return np.random.randn(*shape) * np.sqrt(2.0 / (fan_in + fan_out))
    elif method == 'he':
        return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)
    else:
        raise ValueError(f"Unknown init method: {method}")

class DeepMLP:
    def __init__(self, layer_sizes, activation='tanh', init_method='standard'):
        self.layer_sizes = layer_sizes
        self.activation_fn, self.activation_derivative = ACTIVATIONS[activation]
        self.init_method = init_method
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = init_weights((layer_sizes[i+1], layer_sizes[i]), init_method)
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X
        for i in range(self.num_layers):
            Z = self.weights[i] @ A + self.biases[i]
            self.z_values.append(Z)
            A = self.activation_fn(Z)
            self.activations.append(A)
        return A

    def backward(self, X, y, output):
        m = X.shape[1]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Output layer gradient (MSE loss)
        dA = (output - y) / m
        for i in reversed(range(self.num_layers)):
            Z = self.z_values[i]
            A_prev = self.activations[i]
            dZ = dA * self.activation_derivative(Z)
            grads_w[i] = dZ @ A_prev.T
            grads_b[i] = np.sum(dZ, axis=1, keepdims=True)
            if i > 0:
                dA = self.weights[i].T @ dZ
        return grads_w, grads_b

def run_experiment(depth=10, width=64, activation='tanh', init_method='standard', batch_size=32):
    layer_sizes = [width] * (depth + 1)
    model = DeepMLP(layer_sizes, activation, init_method)
    X = np.random.randn(width, batch_size)
    y = np.random.randn(width, batch_size)
    output = model.forward(X)
    grads_w, _ = model.backward(X, y, output)
    grad_norms = [np.linalg.norm(g, 'fro') for g in grads_w]
    return grad_norms

def main():
    depth = 15
    width = 128
    batch_size = 64

    configs = [
        ('sigmoid', 'standard'),
        ('tanh', 'standard'),
        ('relu', 'standard'),
        ('sigmoid', 'xavier'),
        ('tanh', 'xavier'),
        ('relu', 'he'),
    ]

    results = {}
    for act, init in configs:
        print(f"Running: activation={act}, init={init}")
        norms = run_experiment(depth, width, act, init, batch_size)
        results[f"{act}_{init}"] = norms

    # Plot
    plt.figure(figsize=(12, 6))
    layers = list(range(1, depth + 1))
    for label, norms in results.items():
        plt.semilogy(layers, norms, marker='o', label=label, linewidth=2)
    plt.xlabel('Layer (1 = closest to input)')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title(f'Gradient Norms Across {depth} Layers (Width={width})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('gradient_norms.png', dpi=150)
    print("Plot saved to gradient_norms.png")

    # Print summary statistics
    print("\n--- Gradient Norm Summary (mean ± std) ---")
    for label, norms in results.items():
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        first_last_ratio = norms[0] / norms[-1] if norms[-1] != 0 else np.inf
        print(f"{label:20s}: mean={mean_norm:.2e}, std={std_norm:.2e}, first/last ratio={first_last_ratio:.2e}")

if __name__ == '__main__':
    main()