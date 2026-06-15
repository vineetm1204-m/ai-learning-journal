import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Activation Functions & Derivatives
# ------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def gelu(x):
    # Approximation used in BERT / GPT
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def gelu_grad(x):
    # Derivative of the approximation
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)  # Not exact for approx, but standard Gaussian PDF
    # Exact derivative of the tanh approximation:
    k = np.sqrt(2 / np.pi)
    inner = x + 0.044715 * x ** 3
    tanh_inner = np.tanh(k * inner)
    sech2 = 1 - tanh_inner ** 2
    d_inner = 1 + 3 * 0.044715 * x ** 2
    return cdf + x * 0.5 * sech2 * k * d_inner

# ------------------------------------------------------------
# Experiment 1: Vanishing / Exploding Gradient Simulation
# ------------------------------------------------------------
def simulate_gradient_flow(act_func, grad_func, depth=50, init_scale=1.0):
    """Simulate gradient magnitude through a deep linear chain."""
    x = np.array([1.0])  # dummy input
    grad = np.array([1.0])  # gradient from loss
    magnitudes = []
    
    for _ in range(depth):
        # Forward (not strictly needed for grad magnitude if linear weights=1)
        out = act_func(x)
        # Backward
        grad = grad * grad_func(x)
        magnitudes.append(np.abs(grad).mean())
        # Next layer input (simulate weight=1)
        x = out
    return magnitudes

# ------------------------------------------------------------
# Experiment 2: Dead Neuron Ratio (ReLU vs Leaky ReLU)
# ------------------------------------------------------------
def dead_neuron_ratio(act_func, grad_func, n_neurons=1000, n_samples=100):
    """Percentage of neurons outputting zero gradient for random normal inputs."""
    dead_counts = 0
    for _ in range(n_samples):
        x = np.random.randn(n_neurons)
        g = grad_func(x)
        dead_counts += np.sum(g == 0)
    return dead_counts / (n_neurons * n_samples) * 100

# ------------------------------------------------------------
# Main Execution & Plotting
# ------------------------------------------------------------
if __name__ == "__main__":
    x = np.linspace(-5, 5, 400)
    
    funcs = {
        "Sigmoid": (sigmoid, sigmoid_grad),
        "Tanh": (tanh, tanh_grad),
        "ReLU": (relu, relu_grad),
        "Leaky ReLU (0.01)": (lambda x: leaky_relu(x, 0.01), lambda x: leaky_relu_grad(x, 0.01)),
        "GELU": (gelu, gelu_grad),
    }

    # --- Plot 1: Function Shapes & Gradients ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, (f, g) in funcs.items():
        axes[0].plot(x, f(x), label=name, linewidth=2)
        axes[1].plot(x, g(x), label=name, linewidth=2, linestyle='--')
    
    axes[0].set_title("Activation Functions")
    axes[0].set_ylabel("Output")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Derivatives (Gradients)")
    axes[1].set_xlabel("Input (x)")
    axes[1].set_ylabel("Gradient")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("day2_activations_shapes.png", dpi=150)
    print("Saved: day2_activations_shapes.png")

    # --- Plot 2: Gradient Flow Depth Simulation ---
    plt.figure(figsize=(10, 5))
    depth = 30
    for name, (f, g) in funcs.items():
        mags = simulate_gradient_flow(f, g, depth=depth)
        plt.plot(range(1, depth+1), mags, label=name, marker='o', markersize=3)
    
    plt.yscale('log')
    plt.title(f"Gradient Magnitude vs Network Depth (Log Scale)\n(Simulated Chain: Weight=1, Input=1)")
    plt.xlabel("Layer Depth")
    plt.ylabel("Avg Gradient Magnitude (Log)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("day2_gradient_flow.png", dpi=150)
    print("Saved: day2_gradient_flow.png")

    # --- Experiment 3: Dead Neuron Statistics ---
    print("\n--- Dead Neuron Analysis (1000 neurons, 100 batches) ---")
    for name, (f, g) in funcs.items():
        if name in ["ReLU", "Leaky ReLU (0.01)"]:
            ratio = dead_neuron_ratio(f, g)
            print(f"{name:20s}: {ratio:.2f}% dead gradients (zero derivative)")

    # --- Experiment 4: Output Distribution Shift ---
    print("\n--- Output Statistics for Standard Normal Input ---")
    x_dist = np.random.randn(10000)
    for name, (f, g) in funcs.items():
        out = f(x_dist)
        print(f"{name:20s}: Mean={out.mean():.4f}, Std={out.std():.4f}, "
              f"Min={out.min():.4f}, Max={out.max():.4f}")

    print("\nExperiment complete. Check generated PNG files.")