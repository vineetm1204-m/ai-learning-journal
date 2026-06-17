import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Day 3: Loss Functions Mini-Experiment
# MSE, Cross-Entropy, Huber Loss
# ============================================================

# ------------------------------------------------------------
# 1. Implementations (NumPy, from scratch)
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    """Mean Squared Error: 1/n * sum((y - y_hat)^2)"""
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    """Gradient of MSE w.r.t y_pred: 2/n * (y_pred - y_true)"""
    return 2.0 * (y_pred - y_true) / len(y_true)

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss:
      0.5 * (y - y_hat)^2                  if |y - y_hat| <= delta
      delta * (|y - y_hat| - 0.5 * delta)  otherwise
    """
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)
    return np.mean(np.where(abs_error <= delta, quadratic, linear))

def huber_grad(y_true, y_pred, delta=1.0):
    """Gradient of Huber w.r.t y_pred"""
    error = y_pred - y_true  # note sign flip for grad w.r.t y_pred
    abs_error = np.abs(error)
    grad = np.where(abs_error <= delta, error, delta * np.sign(error))
    return grad / len(y_true)

def softmax(logits):
    """Numerically stable softmax"""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def cross_entropy_loss(y_true, logits):
    """
    Categorical Cross-Entropy with logits (one-hot y_true).
    y_true: (N, C) one-hot
    logits: (N, C) raw scores
    """
    probs = softmax(logits)
    # clip for numerical stability
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return -np.mean(np.sum(y_true * np.log(probs), axis=1))

def cross_entropy_grad(y_true, logits):
    """
    Gradient of CE w.r.t logits: (softmax(logits) - y_true) / N
    """
    probs = softmax(logits)
    return (probs - y_true) / y_true.shape[0]

# ------------------------------------------------------------
# 2. Synthetic Data & Visualization
# ------------------------------------------------------------

def plot_regression_losses():
    """Visualize MSE vs Huber on a 1D regression residual sweep."""
    y_true = 0.0
    y_pred = np.linspace(-3, 3, 400)
    mse_vals = mse_loss(y_true, y_pred)
    huber_vals = huber_loss(y_true, y_pred, delta=1.0)

    plt.figure(figsize=(6, 4))
    plt.plot(y_pred, mse_vals, label="MSE", linewidth=2)
    plt.plot(y_pred, huber_vals, label="Huber (δ=1.0)", linewidth=2, linestyle="--")
    plt.axvline(0, color="gray", linestyle=":", alpha=0.5)
    plt.title("Loss vs Prediction Error (y_true=0)")
    plt.xlabel("Prediction (y_pred)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_classification_loss():
    """Visualize Cross-Entropy vs confidence for correct class."""
    # 3 classes, true class = 0
    y_true = np.array([[1.0, 0.0, 0.0]])
    # Vary logit for class 0, keep others fixed at 0
    logit_range = np.linspace(-5, 5, 200)
    ce_vals = []
    for l in logit_range:
        logits = np.array([[l, 0.0, 0.0]])
        ce_vals.append(cross_entropy_loss(y_true, logits))

    plt.figure(figsize=(6, 4))
    plt.plot(logit_range, ce_vals, label="Cross-Entropy", linewidth=2, color="tab:green")
    plt.axvline(0, color="gray", linestyle=":", alpha=0.5)
    plt.title("Cross-Entropy vs Logit for True Class (others=0)")
    plt.xlabel("Logit for True Class")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 3. Gradient Sanity Checks (Finite Differences)
# ------------------------------------------------------------

def grad_check_scalar(loss_fn, grad_fn, y_true, y_pred, eps=1e-5):
    """Compare analytic gradient to finite difference for scalar output loss."""
    # numeric grad: dL/dy_pred
    loss_plus = loss_fn(y_true, y_pred + eps)
    loss_minus = loss_fn(y_true, y_pred - eps)
    num_grad = (loss_plus - loss_minus) / (2 * eps)
    ana_grad = grad_fn(y_true, y_pred)
    # For scalar y_pred, ana_grad is array; take mean if needed
    if ana_grad.ndim > 0:
        ana_grad = np.mean(ana_grad)
    rel_err = np.abs(num_grad - ana_grad) / (np.abs(num_grad) + np.abs(ana_grad) + 1e-12)
    return num_grad, ana_grad, rel_err

def grad_check_vector(loss_fn, grad_fn, y_true, y_pred, eps=1e-5):
    """Finite-difference check for vector outputs (e.g., logits)."""
    num_grad = np.zeros_like(y_pred)
    it = np.nditer(y_pred, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = y_pred[idx]
        y_pred[idx] = old_val + eps
        loss_plus = loss_fn(y_true, y_pred)
        y_pred[idx] = old_val - eps
        loss_minus = loss_fn(y_true, y_pred)
        y_pred[idx] = old_val
        num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    ana_grad = grad_fn(y_true, y_pred)
    rel_err = np.abs(num_grad - ana_grad) / (np.abs(num_grad) + np.abs(ana_grad) + 1e-12)
    return np.max(rel_err), np.mean(rel_err)

def run_gradient_checks():
    print("=== Gradient Sanity Checks ===")
    # MSE
    y_t = np.array([2.0])
    y_p = np.array([1.5])
    ng, ag, re = grad_check_scalar(mse_loss, mse_grad, y_t, y_p)
    print(f"MSE: num={ng:.6f}, ana={ag:.6f}, rel_err={re:.2e}")

    # Huber
    ng, ag, re = grad_check_scalar(huber_loss, huber_grad, y_t, y_p)
    print(f"Huber: num={ng:.6f}, ana={ag:.6f}, rel_err={re:.2e}")

    # Cross-Entropy (vector)
    y_true = np.array([[0.0, 1.0, 0.0]])  # class 1
    logits = np.array([[0.2, 1.5, -0.5]])
    max_re, mean_re = grad_check_vector(cross_entropy_loss, cross_entropy_grad, y_true, logits.copy())
    print(f"CrossEntropy: max_rel_err={max_re:.2e}, mean_rel_err={mean_re:.2e}")
    print()

# ------------------------------------------------------------
# 4. Mini "Training" Demo: Linear Regression with MSE vs Huber
# ------------------------------------------------------------

def train_linear_regression(loss_fn, grad_fn, X, y, lr=0.05, steps=200, **loss_kwargs):
    """Simple GD on y = w*x + b"""
    w, b = 0.0, 0.0
    losses = []
    for step in range(steps):
        y_pred = w * X + b
        loss = loss_fn(y, y_pred, **loss_kwargs)
        losses.append(loss)
        # grads
        dw = np.mean(grad_fn(y, y_pred, **loss_kwargs) * X)
        db = np.mean(grad_fn(y, y_pred, **loss_kwargs))
        w -= lr * dw
        b -= lr * db
    return w, b, losses

def demo_regression_robustness():
    print("=== Robustness Demo: Clean vs Outlier Data ===")
    np.random.seed(42)
    X = np.linspace(-2, 2, 50)
    y_clean = 2.0 * X + 1.0 + np.random.randn(50) * 0.3

    # Add outliers
    y_outlier = y_clean.copy()
    outlier_idx = [5, 12, 30]
    y_outlier[outlier_idx] += np.array([10.0, -8.0, 12.0])  # large corruptions

    # Train on clean
    w_mse_c, b_mse_c, _ = train_linear_regression(mse_loss, mse_grad, X, y_clean)
    w_hub_c, b_hub_c, _ = train_linear_regression(huber_loss, huber_grad, X, y_clean, delta=1.0)

    # Train on outliers
    w_mse_o, b_mse_o, _ = train_linear_regression(mse_loss, mse_grad, X, y_outlier)
    w_hub_o, b_hub_o, _ = train_linear_regression(huber_loss, huber_grad, X, y_outlier, delta=1.0)

    print(f"Clean Data:      True(w=2.0, b=1.0)")
    print(f"  MSE:           w={w_mse_c:.3f}, b={b_mse_c:.3f}")
    print(f"  Huber:         w={w_hub_c:.3f}, b={b_hub_c:.3f}")
    print(f"Outlier Data:")
    print(f"  MSE:           w={w_mse_o:.3f}, b={b_mse_o:.3f}  <-- heavily pulled")
    print(f"  Huber (δ=1.0): w={w_hub_o:.3f}, b={b_hub_o:.3f}  <-- robust")
    print()

    # Plot
    plt.figure(figsize=(10, 4))
    # Clean
    plt.subplot(1, 2, 1)
    plt.scatter(X, y_clean, alpha=0.6, label="Clean Data")
    plt.plot(X, w_mse_c * X + b_mse_c, 'r-', label=f"MSE fit (w={w_mse_c:.2f})")
    plt.plot(X, w_hub_c * X + b_hub_c, 'g--', label=f"Huber fit (w={w_hub_c:.2f})")
    plt.plot(X, 2*X+1, 'k:', label="True")
    plt.title("Clean Data")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    # Outliers
    plt.subplot(1, 2, 2)
    plt.scatter(X, y_outlier, alpha=0.6, label="Data + Outliers")
    plt.scatter(X[outlier_idx], y_outlier[outlier_idx], color='red', s=80, zorder=5, label="Outliers")
    plt.plot(X, w_mse_o * X + b_mse_o, 'r-', label=f"MSE fit (w={w_mse_o:.2f})")
    plt.plot(X, w_hub_o * X + b_hub_o, 'g--', label=f"Huber fit (w={w_hub_o:.2f})")
    plt.plot(X, 2*X+1, 'k:', label="True")
    plt.title("With Outliers")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 5. Mini "Training" Demo: Softmax Classification with CE
# ------------------------------------------------------------

def train_softmax_classifier(X, y_onehot, lr=0.1, steps=500):
    """Linear softmax classifier: logits = X @ W + b"""
    n_samples, n_features = X.shape
    n_classes = y_onehot.shape[1]
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)
    losses = []
    accs = []
    for step in range(steps):
        logits = X @ W + b
        loss = cross_entropy_loss(y_onehot, logits)
        losses.append(loss)
        # accuracy
        preds = np.argmax(logits, axis=1)
        true_labels = np.argmax(y_onehot, axis=1)
        accs.append(np.mean(preds == true_labels))
        # gradients
        dlogits = cross_entropy_grad(y_onehot, logits)  # (N, C)
        dW = X.T @ dlogits / n_samples
        db = np.mean(dlogits, axis=0)
        W -= lr * dW
        b -= lr * db
    return W, b, losses, accs

def demo_classification():
    print("=== Softmax Classification Demo ===")
    np.random.seed(123)
    # 3-class 2D blobs
    n_per_class = 100
    centers = np.array([[-2, -2], [2, -2], [0, 2.5]])
    X = np.vstack([np.random.randn(n_per_class, 2) + c for c in centers])
    y = np.hstack([np.full(n_per_class, i) for i in range(3)])
    y_onehot = np.eye(3)[y]

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y_onehot = X[idx], y_onehot[idx]

    W, b, losses, accs = train_softmax_classifier(X, y_onehot, lr=0.5, steps=300)

    print(f"Final Loss: {losses[-1]:.4f}, Final Acc: {accs[-1]:.4f}")
    print()

    # Decision boundary plot
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    logits_grid = grid @ W + b
    preds_grid = np.argmax(logits_grid, axis=1).reshape(xx.shape)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, preds_grid, alpha=0.3, cmap=plt.cm.Set1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
    plt.title("Decision Boundaries (Softmax + CE)")
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Cross-Entropy Loss")
    plt.plot(accs, label="Accuracy")
    plt.xlabel("Step"); plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 6. Main Entry
# ------------------------------------------------------------

if __name__ == "__main__":
    run_gradient_checks()
    plot_regression_losses()
    plot_classification_loss()
    demo_regression_robustness()
    demo_classification()
    print("Day 3 experiment complete.")