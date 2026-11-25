import numpy as np
import matplotlib.pyplot as plt

# --- Activation Functions & Derivatives ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def gelu(x):
    # Approximation used in BERT/GPT (tanh version)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_grad(x):
    # Derivative of the tanh approximation
    k = np.sqrt(2 / np.pi)
    x3 = 0.044715 * x**3
    tanh_arg = k * (x + x3)
    tanh_val = np.tanh(tanh_arg)
    sech2 = 1 - tanh_val**2
    return 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * k * (1 + 3 * 0.044715 * x**2)

# --- Experiment Logic ---

def run_experiment():
    print("=" * 60)
    print("Day 125: Activation Function Analysis")
    print("=" * 60)

    # 1. Range & Saturation Check
    x_test = np.array([-10, -5, -1, 0, 1, 5, 10])
    funcs = {
        "Sigmoid": sigmoid,
        "Tanh": tanh,
        "ReLU": relu,
        "LeakyReLU": leaky_relu,
        "GELU": gelu
    }

    print("\n--- Output Range at Extremes ---")
    print(f"{'Input':>6} | " + " | ".join(f"{name:>10}" for name in funcs.keys()))
    print("-" * 70)
    for val in x_test:
        row = f"{val:>6} | "
        row += " | ".join(f"{f(val):>10.4f}" for f in funcs.values())
        print(row)

    # 2. Gradient Flow Analysis (Vanishing/Exploding)
    print("\n--- Gradient Magnitude at Extremes (Vanishing Check) ---")
    grads = {
        "Sigmoid": sigmoid_grad,
        "Tanh": tanh_grad,
        "ReLU": relu_grad,
        "LeakyReLU": leaky_relu_grad,
        "GELU": gelu_grad
    }
    print(f"{'Input':>6} | " + " | ".join(f"{name:>10}" for name in grads.keys()))
    print("-" * 70)
    for val in x_test:
        row = f"{val:>6} | "
        row += " | ".join(f"{g(val):>10.4f}" for g in grads.values())
        print(row)

    # 3. Dead Neuron Simulation (ReLU vs Leaky/GELU)
    print("\n--- Dead Neuron Simulation: % Zero Gradient for Negative Inputs ---")
    neg_inputs = np.random.uniform(-5, 0, 10000)
    for name, g in grads.items():
        zero_grad_pct = np.mean(g(neg_inputs) == 0) * 100
        print(f"{name:>10}: {zero_grad_pct:.2f}% zero gradient (neurons 'dead' for x<0)")

    # 4. Computational Cost (Micro-benchmark)
    print("\n--- Computational Cost (Relative Speed, 1M elements) ---")
    big_x = np.random.randn(1_000_000)
    import time
    for name, f in funcs.items():
        start = time.perf_counter()
        _ = f(big_x)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{name:>10}: {elapsed:.2f} ms")

    # 5. Visualization
    plot_activations()

def plot_activations():
    x = np.linspace(-4, 4, 400)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    plot_data = [
        ("Sigmoid", sigmoid, sigmoid_grad, (-0.1, 1.1), (-0.1, 0.3)),
        ("Tanh", tanh, tanh_grad, (-1.1, 1.1), (-0.1, 1.1)),
        ("ReLU", relu, relu_grad, (-0.5, 4.5), (-0.1, 1.1)),
        ("Leaky ReLU", leaky_relu, leaky_relu_grad, (-0.5, 4.5), (-0.1, 1.1)),
        ("GELU", gelu, gelu_grad, (-0.5, 4.5), (-0.1, 1.1)),
    ]

    for idx, (name, fn, grad_fn, ylim_fn, ylim_grad) in enumerate(plot_data):
        ax = axes[idx]
        y = fn(x)
        dy = grad_fn(x)
        
        ax.plot(x, y, label=f'{name}(x)', color='tab:blue', linewidth=2)
        ax.plot(x, dy, label=f"Grad", color='tab:orange', linestyle='--', linewidth=2)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(name)
        ax.set_ylim(min(ylim_fn[0], ylim_grad[0]), max(ylim_fn[1], ylim_grad[1]))
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[5])
    
    fig.suptitle("Day 125: Activation Functions & Gradients", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("day125_activations.png", dpi=150)
    print("\nPlot saved to 'day125_activations.png'")
    # plt.show() # Commented out for headless environments

if __name__ == "__main__":
    run_experiment()