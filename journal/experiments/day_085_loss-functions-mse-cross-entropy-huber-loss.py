import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Loss Function Implementations (NumPy)
# ==========================================

def mse_loss(y_true, y_pred):
    """Mean Squared Error: L = mean((y - y_hat)^2)"""
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    """Mean Absolute Error: L = mean(|y - y_hat|)"""
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss: Quadratic for small errors, linear for large errors.
    L = 0.5 * (y - y_hat)^2                  if |y - y_hat| <= delta
        delta * (|y - y_hat| - 0.5 * delta)  otherwise
    """
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * quadratic**2 + delta * linear)

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Binary Cross Entropy: L = -[y log(p) + (1-y) log(1-p)]
    Clips predictions to avoid log(0).
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Categorical Cross Entropy (One-Hot): L = -sum(y_i log(p_i))
    y_true: (N, C) one-hot encoded
    y_pred: (N, C) probabilities (softmax output)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # Sum over classes, mean over batch
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# ==========================================
# 2. Experiment: Regression Loss Landscape
# ==========================================
def run_regression_experiment():
    print("=" * 60)
    print("DAY 85: REGRESSION LOSS LANDSCAPES (Target = 0.0)")
    print("=" * 60)
    
    # Predictions ranging from -3 to 3
    y_pred = np.linspace(-3, 3, 200)
    y_true = np.zeros_like(y_pred) # Target is 0
    
    # Compute losses
    losses = {
        "MSE": mse_loss(y_true, y_pred),
        "MAE": mae_loss(y_true, y_pred),
        "Huber (δ=1.0)": huber_loss(y_true, y_pred, delta=1.0),
        "Huber (δ=0.5)": huber_loss(y_true, y_pred, delta=0.5),
    }
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for name, loss_vals in losses.items():
        plt.plot(y_pred, loss_vals, label=name, linewidth=2)
    
    plt.title("Regression Loss Functions vs. Prediction Error (Target=0)")
    plt.xlabel("Prediction Error (y_pred - y_true)")
    plt.ylabel("Loss Value")
    plt.axvline(0, color='black', linestyle=':', alpha=0.5)
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 5)
    plt.tight_layout()
    plt.savefig("day85_regression_losses.png", dpi=150)
    print("Saved plot: day85_regression_losses.png")
    plt.close()
    
    # Numerical comparison at specific error points
    print("\nLoss Values at Specific Errors:")
    print(f"{'Error':>8} | {'MSE':>8} | {'MAE':>8} | {'Huber(1.0)':>10} | {'Huber(0.5)':>10}")
    print("-" * 55)
    for err in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]:
        y_p = np.array([err])
        y_t = np.array([0.0])
        print(f"{err:>8.2f} | {mse_loss(y_t, y_p):>8.4f} | {mae_loss(y_t, y_p):>8.4f} | "
              f"{huber_loss(y_t, y_p, 1.0):>10.4f} | {huber_loss(y_t, y_p, 0.5):>10.4f}")

# ==========================================
# 3. Experiment: Classification Loss Landscape
# ==========================================
def run_classification_experiment():
    print("\n" + "=" * 60)
    print("DAY 85: CLASSIFICATION LOSS LANDSCAPES (Binary)")
    print("=" * 60)
    
    # Predicted probability for Class 1 (True label = 1)
    p = np.linspace(0.001, 0.999, 200)
    y_true = np.ones_like(p)
    
    bce = binary_cross_entropy(y_true, p)
    # MSE applied to classification (for comparison)
    mse = mse_loss(y_true, p)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p, bce, label="Binary Cross Entropy", linewidth=2, color='red')
    plt.plot(p, mse, label="MSE (on probabilities)", linewidth=2, linestyle='--', color='blue')
    
    plt.title("Classification Loss vs. Predicted Probability (True Label = 1)")
    plt.xlabel("Predicted Probability P(y=1)")
    plt.ylabel("Loss Value")
    plt.axvline(1.0, color='green', linestyle=':', label='Perfect Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig("day85_classification_losses.png", dpi=150)
    print("Saved plot: day85_classification_losses.png")
    plt.close()
    
    print("\nLoss Values at Specific Probabilities (True Label = 1):")
    print(f"{'P(y=1)':>10} | {'BCE':>10} | {'MSE':>10}")
    print("-" * 35)
    for prob in [0.01, 0.1, 0.5, 0.9, 0.99]:
        y_p = np.array([prob])
        y_t = np.array([1.0])
        print(f"{prob:>10.2f} | {binary_cross_entropy(y_t, y_p):>10.4f} | {mse_loss(y_t, y_p):>10.4f}")

# ==========================================
# 4. Gradient Analysis (Numerical)
# ==========================================
def gradient_analysis():
    print("\n" + "=" * 60)
    print("DAY 85: GRADIENT BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    def numerical_grad(func, y_true, y_pred, h=1e-5):
        return (func(y_true, y_pred + h) - func(y_true, y_pred - h)) / (2 * h)
    
    y_true = np.array([0.0])
    errors = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    
    print(f"{'Error':>8} | {'MSE Grad':>10} | {'MAE Grad':>10} | {'Huber(1) Grad':>12}")
    print("-" * 48)
    for err in errors:
        y_pred = np.array([err])
        g_mse = numerical_grad(mse_loss, y_true, y_pred)
        g_mae = numerical_grad(mae_loss, y_true, y_pred)
        g_huber = numerical_grad(lambda t, p: huber_loss(t, p, 1.0), y_true, y_pred)
        print(f"{err:>8.1f} | {g_mse:>10.4f} | {g_mae:>10.4f} | {g_huber:>12.4f}")
    
    print("\nObservation: MSE gradient grows linearly (explodes for outliers).")
    print("             MAE gradient is constant (sign).")
    print("             Huber transitions from linear to constant at delta.")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    
    run_regression_experiment()
    run_classification_experiment()
    gradient_analysis()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE: Check generated PNG files.")
    print("=" * 60)