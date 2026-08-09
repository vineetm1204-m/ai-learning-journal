import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Loss Function Implementations (from scratch)
# ============================================================

def mse_loss(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    """Gradient of MSE w.r.t y_pred"""
    return 2 * (y_pred - y_true) / len(y_true)

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss: quadratic for small errors, linear for large"""
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * quadratic ** 2 + delta * linear)

def huber_grad(y_true, y_pred, delta=1.0):
    """Gradient of Huber loss w.r.t y_pred"""
    error = y_pred - y_true
    abs_error = np.abs(error)
    grad = np.where(abs_error <= delta, error, delta * np.sign(error))
    return grad / len(y_true)

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """Binary Cross-Entropy (log loss)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_grad(y_true, y_pred, eps=1e-15):
    """Gradient of BCE w.r.t y_pred (before sigmoid)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))

def categorical_cross_entropy(y_true, y_pred, eps=1e-15):
    """Categorical Cross-Entropy for one-hot targets"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cce_grad(y_true, y_pred, eps=1e-15):
    """Gradient of CCE w.r.t logits (before softmax)"""
    return (y_pred - y_true) / len(y_true)

# ============================================================
# Experiment 1: Loss Landscapes (Regression)
# ============================================================

def plot_regression_losses():
    y_true = 0.0
    y_pred = np.linspace(-3, 3, 400)
    
    mse = np.array([mse_loss(y_true, yp) for yp in y_pred])
    huber_05 = np.array([huber_loss(y_true, yp, delta=0.5) for yp in y_pred])
    huber_1 = np.array([huber_loss(y_true, yp, delta=1.0) for yp in y_pred])
    huber_2 = np.array([huber_loss(y_true, yp, delta=2.0) for yp in y_pred])
    
    mse_g = np.array([mse_grad(y_true, yp) for yp in y_pred])
    huber_1_g = np.array([huber_grad(y_true, yp, delta=1.0) for yp in y_pred])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss values
    ax = axes[0, 0]
    ax.plot(y_pred, mse, label='MSE', linewidth=2)
    ax.plot(y_pred, huber_05, label='Huber (δ=0.5)', linewidth=2)
    ax.plot(y_pred, huber_1, label='Huber (δ=1.0)', linewidth=2)
    ax.plot(y_pred, huber_2, label='Huber (δ=2.0)', linewidth=2)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Prediction (y_pred)')
    ax.set_ylabel('Loss')
    ax.set_title('Regression Loss Functions (target=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gradients
    ax = axes[0, 1]
    ax.plot(y_pred, mse_g, label='MSE gradient', linewidth=2)
    ax.plot(y_pred, huber_1_g, label='Huber gradient (δ=1.0)', linewidth=2)
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Prediction (y_pred)')
    ax.set_ylabel('Gradient')
    ax.set_title('Gradients w.r.t Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Zoom near zero
    ax = axes[1, 0]
    mask = np.abs(y_pred) < 1.5
    ax.plot(y_pred[mask], mse[mask], label='MSE', linewidth=2)
    ax.plot(y_pred[mask], huber_1[mask], label='Huber (δ=1.0)', linewidth=2)
    ax.set_xlabel('Prediction (y_pred)')
    ax.set_ylabel('Loss')
    ax.set_title('Zoom: Quadratic Region (|error| < δ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale for large errors
    ax = axes[1, 1]
    ax.semilogy(np.abs(y_pred), mse, label='MSE', linewidth=2)
    ax.semilogy(np.abs(y_pred), huber_1, label='Huber (δ=1.0)', linewidth=2)
    ax.set_xlabel('|Error|')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Asymptotic Behavior (Large Errors)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day44_regression_losses.png', dpi=150)
    plt.close()

# ============================================================
# Experiment 2: Classification Loss Landscapes
# ============================================================

def plot_classification_losses():
    # Binary classification: target = 1
    p = np.linspace(0.001, 0.999, 400)
    y_true = 1.0
    
    bce = - (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    mse_cls = (y_true - p) ** 2
    
    # Gradients w.r.t probability
    bce_g = (p - y_true) / (p * (1 - p))
    mse_g = 2 * (p - y_true)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(p, bce, label='Binary Cross-Entropy', linewidth=2)
    ax.plot(p, mse_cls, label='MSE', linewidth=2)
    ax.axvline(1, color='k', linestyle=':', alpha=0.3, label='Target (y=1)')
    ax.set_xlabel('Predicted Probability (p)')
    ax.set_ylabel('Loss')
    ax.set_title('Binary Classification Losses (target=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)
    
    ax = axes[0, 1]
    ax.plot(p, bce_g, label='BCE gradient', linewidth=2)
    ax.plot(p, mse_g, label='MSE gradient', linewidth=2)
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(1, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Predicted Probability (p)')
    ax.set_ylabel('Gradient')
    ax.set_title('Gradients w.r.t Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 10)
    
    # Categorical: 3 classes, target = class 0
    logits = np.linspace(-3, 3, 200)
    # Softmax for 3 classes where logit_0 varies, others fixed at 0
    def softmax_3(logit_0):
        l = np.array([logit_0, 0.0, 0.0])
        e = np.exp(l - np.max(l))
        return e / e.sum()
    
    probs = np.array([softmax_3(l) for l in logits])
    cce = -np.log(probs[:, 0])
    mse_cat = np.mean((np.array([1,0,0]) - probs) ** 2, axis=1)
    
    ax = axes[1, 0]
    ax.plot(logits, cce, label='Categorical CE', linewidth=2)
    ax.plot(logits, mse_cat, label='MSE', linewidth=2)
    ax.set_xlabel('Logit for True Class (others=0)')
    ax.set_ylabel('Loss')
    ax.set_title('Categorical Classification (3 classes, target=class 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gradient w.r.t logits
    cce_g = probs[:, 0] - 1  # dL/dlogit = p - y
    mse_g_cat = 2 * (probs[:, 0] - 1) * probs[:, 0] * (1 - probs[:, 0])  # chain rule
    
    ax = axes[1, 1]
    ax.plot(logits, cce_g, label='CCE gradient', linewidth=2)
    ax.plot(logits, mse_g_cat, label='MSE gradient', linewidth=2)
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Logit for True Class')
    ax.set_ylabel('Gradient')
    ax.set_title('Gradients w.r.t Logits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day44_classification_losses.png', dpi=150)
    plt.close()

# ============================================================
# Experiment 3: Outlier Robustness Demo
# ============================================================

def outlier_robustness_demo():
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y_true = 2 * x + 1 + np.random.randn(n) * 0.5
    
    # Add outliers
    outlier_idx = [10, 30, 50, 70, 90]
    y_outlier = y_true.copy()
    y_outlier[outlier_idx] += np.random.randn(len(outlier_idx)) * 10
    
    # Fit with MSE (analytical solution for linear regression)
    X = np.column_stack([x, np.ones_like(x)])
    
    # MSE solution
    theta_mse = np.linalg.lstsq(X, y_outlier, rcond=None)[0]
    y_pred_mse = X @ theta_mse
    
    # Huber solution via iteratively reweighted least squares (IRLS)
    theta_huber = np.linalg.lstsq(X, y_outlier, rcond=None)[0]
    for _ in range(20):
        residuals = y_outlier - X @ theta_huber
        weights = np.where(np.abs(residuals) <= 1.0, 1.0, 1.0 / np.abs(residuals))
        W = np.diag(weights)
        theta_huber = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y_outlier)
    y_pred_huber = X @ theta_huber
    
    # Clean fit (oracle)
    theta_clean = np.linalg.lstsq(X, y_true, rcond=None)[0]
    y_pred_clean = X @ theta_clean
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, y_data, y_pred, title, theta in zip(
        axes,
        [y_outlier, y_outlier, y_outlier],
        [y_pred_mse, y_pred_huber, y_pred_clean],
        ['MSE Fit (sensitive to outliers)', 'Huber Fit (robust)', 'Oracle (clean data)'],
        [theta_mse, theta_huber, theta_clean]
    ):
        ax.scatter(x, y_data, alpha=0.5, s=20, label='Data (with outliers)')
        ax.scatter(x[outlier_idx], y_data[outlier_idx], color='red', s=50, zorder=5, label='Outliers')
        ax.plot(x, y_pred, 'r-', linewidth=2, label=f'Fit: y={theta[0]:.2f}x+{theta[1]:.2f}')
        ax.plot(x, 2*x+1, 'k--', alpha=0.5, label='True: y=2x+1')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day44_outlier_robustness.png', dpi=150)
    plt.close()
    
    print("Outlier Robustness Results:")
    print(f"  True params:      slope=2.00, intercept=1.00")
    print(f"  MSE fit:          slope={theta_mse[0]:.4f}, intercept={theta_mse[1]:.4f}")
    print(f"  Huber fit:        slope={theta_huber[0]:.4f}, intercept={theta_huber[1]:.4f}")
    print(f"  Oracle (clean):   slope={theta_clean[0]:.4f}, intercept={theta_clean[1]:.4f}")

# ============================================================
# Experiment 4: Gradient Behavior in Training Dynamics
# ============================================================

def training_dynamics_demo():
    """Compare convergence on a simple 1D regression with outliers"""
    np.random.seed(123)
    n = 50
    x = np.random.randn(n)
    y = 3 * x + 2 + np.random.randn(n) * 0.3
    
    # Add a few large outliers
    outlier_mask = np.random.choice(n, 3, replace=False)
    y[outlier_mask] += np.random.randn(3) * 15
    
    # Simple gradient descent
    def train(loss_fn, grad_fn, lr=0.01, steps=500, delta=None):
        w, b = 0.0, 0.0
        history = []
        for step in range(steps):
            y_pred = w * x + b
            if delta is not None:
                loss = loss_fn(y, y_pred, delta=delta)
                grad_w = np.mean(grad_fn(y, y_pred, delta=delta) * x)
                grad_b = np.mean(grad_fn(y, y_pred, delta=delta))
            else:
                loss = loss_fn(y, y_pred)
                grad_w = np.mean(grad_fn(y, y_pred) * x)
                grad_b = np.mean(grad_fn(y, y_pred))
            w -= lr * grad_w
            b -= lr * grad_b
            history.append((loss, w, b))
        return np.array(history)
    
    hist_mse = train(mse_loss, mse_grad, lr=0.05, steps=300)
    hist_huber = train(huber_loss, huber_grad, lr=0.05, steps=300, delta=1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(hist_mse[:, 0], label='MSE', linewidth=2)
    ax.plot(hist_huber[:, 0], label='Huber (δ=1.0)', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Parameter trajectories
    ax = axes[0, 1]
    ax.plot(hist_mse[:, 1], hist_mse[:, 2], 'b-', alpha=0.7, label='MSE trajectory')
    ax.plot(hist_huber[:, 1], hist_huber[:, 2], 'r-', alpha=0.7, label='Huber trajectory')
    ax.scatter([hist_mse[0,1], hist_huber[0,1]], [hist_mse[0,2], hist_huber[0,2]], 
               color=['blue','red'], s=50, zorder=5, label='Start')
    ax.scatter([hist_mse[-1,1], hist_huber[-1,1]], [hist_mse[-1,2], hist_huber[-1,2]], 
               color=['blue','red'], marker='*', s=200, zorder=5, label='End')
    ax.scatter([3], [2], color='black', marker='x', s=100, label='True (3, 2)')
    ax.set_xlabel('Weight (w)')
    ax.set_ylabel('Bias (b)')
    ax.set_title('Parameter Space Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final fits
    ax = axes[1, 0]
    ax.scatter(x, y, alpha=0.5, s=30, label='Data')
    ax.scatter(x[outlier_mask], y[outlier_mask], color='red', s=60, zorder=5, label='Outliers')
    x_line = np.linspace(-3, 3, 100)
    ax.plot(x_line, hist_mse[-1,1]*x_line + hist_mse[-1,2], 'b-', linewidth=2, 
            label=f'MSE: y={hist_mse[-1,1]:.2f}x+{hist_mse[-1,2]:.2f}')
    ax.plot(x_line, hist_huber[-1,1]*x_line + hist_huber[-1,2], 'r-', linewidth=2,
            label=f'Huber: y={hist_huber[-1,1]:.2f}x+{hist_huber[-1,2]:.2f}')
    ax.plot(x_line, 3*x_line + 2, 'k--', alpha=0.5, label='True: y=3x+2')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Final Fits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gradient magnitudes over training
    ax = axes[1, 1]
    mse_grad_mag = []
    huber_grad_mag = []
    for i in range(len(hist_mse)):
        y_pred_mse = hist_mse[i,1] * x + hist_mse[i,2]
        y_pred_huber = hist_huber[i,1] * x + hist_huber[i,2]
        g_mse = mse_grad(y, y_pred_mse)
        g_huber = huber_grad(y, y_pred_huber, delta=1.0)
        mse_grad_mag.append(np.mean(np.abs(g_mse)))
        huber_grad_mag.append(np.mean(np.abs(g_huber)))
    ax.plot(mse_grad_mag, label='MSE avg |grad|', linewidth=2)
    ax.plot(huber_grad_mag, label='Huber avg |grad|', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Average |Gradient|')
    ax.set_title('Gradient Magnitude During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day44_training_dynamics.png', dpi=150)
    plt.close()
    
    print("\nTraining Dynamics Results:")
    print(f"  MSE final:      w={hist_mse[-1,1]:.4f}, b={hist_mse[-1,2]:.4f}, loss={hist_mse[-1,0]:.4f}")
    print(f"  Huber final:    w={hist_huber[-1,1]:.4f}, b={hist_huber[-1,2]:.4f}, loss={hist_huber[-1,0]:.4f}")
    print(f"  True:           w=3.0000, b=2.0000")

# ============================================================
# Experiment 5: When to Use Which Loss - Decision Guide
# ============================================================

def print_decision_guide():
    guide = """
====================================================================
                    LOSS FUNCTION DECISION GUIDE
====================================================================

REGRESSION TASKS:
-----------------
✓ MSE (L2 Loss)
  - Default choice for regression
  - Assumes Gaussian noise
  - Sensitive to outliers (quadratic penalty)
  - Smooth gradients everywhere
  - Use when: errors are normally distributed, no significant outliers

✓ MAE (L1 Loss) - not implemented here but worth knowing
  - Linear penalty, constant gradient magnitude
  - More robust to outliers than MSE
  - Non-differentiable at zero
  - Use when: Laplace noise, want median prediction

✓ Huber Loss (Smooth L1)
  - Best of both worlds: quadratic near zero, linear for large errors
  - Controlled by δ (delta): transition point
  - Differentiable everywhere
  - Use when: outliers present but want smooth optimization
  - δ ≈ 1.0 is common default; tune based on noise scale

✓ Log-Cosh Loss
  - log(cosh(x)) ≈ x²/2 for small x, ≈ |x| - log(2) for large x
  - Smooth alternative to Huber
  - Use when: want smooth, robust loss without hyperparameter

CLASSIFICATION TASKS:
---------------------
✓ Binary Cross-Entropy (Log Loss)
  - Default for binary classification
  - Matches Bernoulli likelihood
  - Strong gradients for confident wrong predictions
  - Use with sigmoid output

✓ Categorical Cross-Entropy
  - Multi-class classification (mutually exclusive classes)
  - Matches Categorical likelihood
  - Use with softmax output
  - Labels: one-hot encoded

✓ Sparse Categorical Cross-Entropy
  - Same as CCE but labels as integers (memory efficient)
  - Use when: many classes, don't want one-hot

✓ Focal Loss
  - Modulates CE to focus on hard examples
  - (1-p_t)^γ * CE
  - Use when: extreme class imbalance (e.g., object detection)

✓ Label Smoothing Cross-Entropy
  - Prevents overconfidence, improves calibration
  - Use when: want better calibrated probabilities

KEY INSIGHTS:
-------------
1. MSE ↔ BCE: Both are maximum likelihood under Gaussian/Bernoulli noise
2. Huber interpolates between MSE and MAE
3. For classification, CE provides stronger gradients when wrong & confident
4. MSE on probabilities saturates (small gradients near 0/1) - avoid!
5. Always match loss to output activation: sigmoid→BCE, softmax→CCE, linear→MSE/Huber

PRACTICAL TIPS:
---------------
- Start with defaults: MSE (regression), BCE/CCE (classification)
- If outliers hurt regression → try Huber (δ=1.0) or MAE
- If class imbalance → try class weights or Focal Loss
- If overconfident predictions → Label Smoothing
- Monitor gradient magnitudes: exploding/vanishing hints at wrong loss
====================================================================
"""
    print(guide)

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 44: Loss Functions Mini-Experiment")
    print("=" * 60)
    
    print("\n[1/5] Plotting regression loss landscapes...")
    plot_regression_losses()
    print("    Saved: day44_regression_losses.png")
    
    print("\n[2/5] Plotting classification loss landscapes...")
    plot_classification_losses()
    print("    Saved: day44_classification_losses.png")
    
    print("\n[3/5] Running outlier robustness demo...")
    outlier_robustness_demo()
    print("    Saved: day44_outlier_robustness.png")
    
    print("\n[4/5] Running training dynamics comparison...")
    training_dynamics_demo()
    print("    Saved: day44_training_dynamics.png")
    
    print("\n[5/5] Printing decision guide...")
    print_decision_guide()
    
    print("\n" + "=" * 60)
    print("Experiment complete! Check generated PNG files.")
    print("=" * 60)