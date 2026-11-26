import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Day 126: Loss Function Landscape Exploration
# MSE, Cross-Entropy (Binary), Huber Loss
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    """Mean Squared Error: L = mean((y - y_hat)^2)"""
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    """Gradient dL/dy_hat = 2 * (y_hat - y) / N"""
    return 2.0 * (y_pred - y_true) / len(y_true)

def bce_loss(y_true, y_pred, eps=1e-15):
    """Binary Cross-Entropy: L = -mean(y*log(p) + (1-y)*log(1-p))"""
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_grad(y_true, y_pred, eps=1e-15):
    """Gradient dL/dy_hat = (y_hat - y) / (y_hat * (1 - y_hat)) / N"""
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss: Quadratic for |e|<=delta, Linear otherwise."""
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)
    return np.mean(np.where(abs_error <= delta, quadratic, linear))

def huber_grad(y_true, y_pred, delta=1.0):
    """Gradient dL/dy_hat."""
    error = y_pred - y_true
    abs_error = np.abs(error)
    grad = np.where(abs_error <= delta, error, delta * np.sign(error))
    return grad / len(y_true)

# ------------------------------------------------------------
# Experiment 1: 1D Loss Landscape Visualization
# ------------------------------------------------------------
def plot_1d_landscape():
    y_true = 0.0
    y_pred_range = np.linspace(-3, 3, 400)

    losses = {
        "MSE": mse_loss(y_true, y_pred_range),
        "Huber (δ=1.0)": huber_loss(y_true, y_pred_range, delta=1.0),
        "Huber (δ=0.5)": huber_loss(y_true, y_pred_range, delta=0.5),
    }

    # BCE requires y_true in {0, 1} and y_pred in (0, 1)
    # We simulate a "probability" target for visualization
    y_true_bce = 1.0
    y_pred_bce = np.linspace(0.01, 0.99, 400)
    losses["BCE (target=1)"] = bce_loss(y_true_bce, y_pred_bce)

    plt.figure(figsize=(10, 6))
    for name, loss_vals in losses.items():
        if name == "BCE (target=1)":
            plt.plot(y_pred_bce, loss_vals, label=name, linestyle='--')
        else:
            plt.plot(y_pred_range, loss_vals, label=name)

    plt.axvline(0, color='k', linestyle=':', alpha=0.3, label='Target (Regression)')
    plt.axvline(1, color='r', linestyle=':', alpha=0.3, label='Target (Classification)')
    plt.ylim(0, 5)
    plt.xlabel("Prediction ($\\hat{y}$)")
    plt.ylabel("Loss")
    plt.title("Day 126: Loss Function Landscapes (1D)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("day126_loss_landscape.png", dpi=150)
    print("Saved landscape plot to day126_loss_landscape.png")
    plt.close()

# ------------------------------------------------------------
# Experiment 2: Gradient Behavior Comparison
# ------------------------------------------------------------
def plot_gradients():
    y_true = 0.0
    y_pred_range = np.linspace(-3, 3, 400)

    grads = {
        "MSE Grad": mse_grad(y_true, y_pred_range),
        "Huber Grad (δ=1.0)": huber_grad(y_true, y_pred_range, delta=1.0),
        "Huber Grad (δ=0.5)": huber_grad(y_true, y_pred_range, delta=0.5),
    }

    plt.figure(figsize=(10, 5))
    for name, grad_vals in grads.items():
        plt.plot(y_pred_range, grad_vals, label=name)

    plt.axhline(0, color='k', alpha=0.3)
    plt.axvline(0, color='k', alpha=0.3)
    plt.xlabel("Prediction ($\\hat{y}$)")
    plt.ylabel("Gradient ($\\partial L / \\partial \\hat{y}$)")
    plt.title("Day 126: Gradient Magnitude vs Prediction Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("day126_gradients.png", dpi=150)
    print("Saved gradient plot to day126_gradients.png")
    plt.close()

# ------------------------------------------------------------
# Experiment 3: Robustness to Outliers (Mini SGD Demo)
# ------------------------------------------------------------
def sgd_robustness_demo():
    np.random.seed(42)
    # True parameter
    w_true = 2.0
    # Clean data
    X = np.random.randn(100, 1)
    y_clean = w_true * X.squeeze() + 0.1 * np.random.randn(100)
    # Corrupt 10% with massive outliers
    y_corrupt = y_clean.copy()
    outlier_idx = np.random.choice(100, 10, replace=False)
    y_corrupt[outlier_idx] += np.random.uniform(20, 50, 10) * np.random.choice([-1, 1], 10)

    losses_to_test = {
        "MSE": (mse_loss, mse_grad),
        "Huber (δ=1.0)": (huber_loss, huber_grad),
    }

    results = {}
    lr = 0.05
    epochs = 200

    for name, (loss_fn, grad_fn) in losses_to_test.items():
        w = 0.0
        history = []
        for _ in range(epochs):
            preds = w * X.squeeze()
            loss = loss_fn(y_corrupt, preds)
            grad = np.mean(grad_fn(y_corrupt, preds) * X.squeeze())
            w -= lr * grad
            history.append((w, loss))
        results[name] = history

    plt.figure(figsize=(12, 5))

    # Parameter convergence
    plt.subplot(1, 2, 1)
    for name, hist in results.items():
        ws = [h[0] for h in hist]
        plt.plot(ws, label=f"{name} (Final w={ws[-1]:.3f})")
    plt.axhline(w_true, color='k', linestyle='--', label=f"True w={w_true}")
    plt.xlabel("Epoch")
    plt.ylabel("Weight Estimate")
    plt.title("SGD Convergence with 10% Outliers")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss curves
    plt.subplot(1, 2, 2)
    for name, hist in results.items():
        ls = [h[1] for h in hist]
        plt.plot(ls, label=name)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("day126_robustness.png", dpi=150)
    print("Saved robustness demo to day126_robustness.png")
    plt.close()

    # Print final weights
    print("\n--- Robustness Summary ---")
    for name, hist in results.items():
        print(f"{name:20s} -> Final Weight: {hist[-1][0]:.4f} (Error: {abs(hist[-1][0] - w_true):.4f})")

# ------------------------------------------------------------
# Experiment 4: BCE vs MSE for Classification (Logits vs Probs)
# ------------------------------------------------------------
def classification_loss_comparison():
    # Target is class 1 (probability 1.0)
    y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # Predictions: confident correct, unconfident correct, wrong
    preds = np.array([0.9, 0.51, 0.99, 0.1, 0.01])

    print("\n--- Classification Loss Comparison (Target=1) ---")
    print(f"{'Pred':>6} | {'MSE':>8} | {'BCE':>8} | {'Log(BCE)':>10}")
    print("-" * 40)
    for p in preds:
        mse = mse_loss(np.array([1.0]), np.array([p]))
        bce = bce_loss(np.array([1.0]), np.array([p]))
        print(f"{p:6.2f} | {mse:8.4f} | {bce:8.4f} | {np.log(bce+1e-9):10.4f}")

    # Gradient magnitude for a wrong prediction (p=0.01)
    print("\nGradient Magnitude for Wrong Prediction (Target=1, Pred=0.01):")
    print(f"  MSE Grad: {mse_grad(np.array([1.0]), np.array([0.01]))[0]:.4f}")
    print(f"  BCE Grad: {bce_grad(np.array([1.0]), np.array([0.01]))[0]:.4f}")

# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Day 126: Loss Functions Mini-Experiment")
    print("=" * 60)

    plot_1d_landscape()
    plot_gradients()
    sgd_robustness_demo()
    classification_loss_comparison()

    print("\nAll experiments complete. Check generated PNG files.")