import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. SYNTHETIC DATASET GENERATION
# ==========================================
def generate_data(n_samples=1000, n_features=10, noise=0.1, seed=42):
    """Generate a linear regression problem: y = X @ w_true + b_true + noise"""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    b_true = 5.0
    y = X @ w_true + b_true + noise * np.random.randn(n_samples)
    # Add bias column to X for vectorized implementation
    X_b = np.c_[X, np.ones(n_samples)]
    return X_b, y, np.append(w_true, b_true)

# ==========================================
# 2. LOSS & GRADIENT FUNCTIONS
# ==========================================
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradient(X_batch, y_batch, w):
    """Gradient of MSE: -2/N * X^T @ (y - Xw)"""
    n = X_batch.shape[0]
    preds = X_batch @ w
    error = preds - y_batch
    grad = (2.0 / n) * (X_batch.T @ error)
    return grad

# ==========================================
# 3. OPTIMIZER IMPLEMENTATIONS
# ==========================================
def run_gd(X, y, lr=0.01, epochs=50, batch_size=None, verbose=False):
    """
    Generic Gradient Descent Runner.
    batch_size=None  -> Batch GD (full dataset)
    batch_size=1     -> Stochastic GD (SGD)
    batch_size=k     -> Mini-batch GD
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    loss_history = []
    time_history = []
    start_time = time.time()

    if batch_size is None:
        batch_size = n_samples # Batch GD

    n_batches = int(np.ceil(n_samples / batch_size))

    for epoch in range(epochs):
        # Shuffle data every epoch (crucial for SGD/Mini-batch)
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0.0

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 1. Compute Gradient
            grad = compute_gradient(X_batch, y_batch, w)

            # 2. Update Weights
            w -= lr * grad

            # Track batch loss (optional, for verbose)
            epoch_loss += mse_loss(y_batch, X_batch @ w) * (end - start)

        avg_epoch_loss = epoch_loss / n_samples
        loss_history.append(avg_epoch_loss)
        time_history.append(time.time() - start_time)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d} | Loss: {avg_epoch_loss:.6f} | Time: {time_history[-1]:.4f}s")

    return w, loss_history, time_history

# ==========================================
# 4. EXPERIMENT RUNNER
# ==========================================
def main():
    print("="*60)
    print("DAY 4: GRADIENT DESCENT VARIANTS COMPARISON")
    print("="*60)

    # Config
    N_SAMPLES = 5000
    N_FEATURES = 20
    EPOCHS = 100
    LR = 0.05

    # Data
    X, y, w_true = generate_data(N_SAMPLES, N_FEATURES, noise=0.5)
    print(f"Data: {N_SAMPLES} samples, {N_FEATURES} features.")
    print(f"True Optimal Loss (Noise floor): ~{0.5**2:.4f}\n")

    # --- Run Variants ---
    configs = {
        "Batch GD":       {"batch_size": None, "lr": LR, "epochs": EPOCHS},
        "Mini-batch (32)": {"batch_size": 32,   "lr": LR, "epochs": EPOCHS},
        "Mini-batch (256)": {"batch_size": 256,  "lr": LR, "epochs": EPOCHS},
        "SGD (Batch=1)":   {"batch_size": 1,    "lr": LR * 0.1, "epochs": EPOCHS}, # SGD often needs smaller LR
    }

    results = {}

    for name, cfg in configs.items():
        print(f"\n--- Running {name} ---")
        w_final, losses, times = run_gd(X, y, verbose=True, **cfg)
        results[name] = {"losses": losses, "times": times, "weights": w_final}
        final_loss = losses[-1]
        dist_to_opt = np.linalg.norm(w_final - w_true)
        print(f"Final Loss: {final_loss:.6f} | Param Dist to True: {dist_to_opt:.4f}")

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Day 4: Gradient Descent Variants Comparison", fontsize=14)

    # Plot 1: Loss vs Epochs
    ax = axes[0]
    for name, res in results.items():
        ax.plot(res["losses"], label=name, alpha=0.8, linewidth=1.5)
    ax.axhline(y=0.25, color='k', linestyle='--', label='Noise Floor (Theoretical Min)')
    ax.set_yscale('log')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss (Log Scale)")
    ax.set_title("Convergence Speed (Loss vs Epochs)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 2: Loss vs Wall-clock Time
    ax = axes[1]
    for name, res in results.items():
        ax.plot(res["times"], res["losses"], label=name, alpha=0.8, linewidth=1.5)
    ax.axhline(y=0.25, color='k', linestyle='--', label='Noise Floor')
    ax.set_yscale('log')
    ax.set_xlabel("Wall-clock Time (seconds)")
    ax.set_ylabel("MSE Loss (Log Scale)")
    ax.set_title("Computational Efficiency (Loss vs Time)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 6. ANALYSIS SUMMARY
    # ==========================================
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print("1. BATCH GD: Smooth convergence, stable. Computationally heavy per epoch.")
    print("2. MINI-BATCH (32): Noisy path but fast initial progress. Best trade-off usually.")
    print("3. MINI-BATCH (256): Smoother than 32, faster per epoch than Batch GD.")
    print("4. SGD (1): Highly noisy trajectory. Requires lower LR. Slow convergence in")
    print("   terms of epochs, but each step is extremely fast. Good for massive datasets.")
    print("\nKey Insight: Mini-batch (32-256) typically dominates in wall-clock time")
    print("for non-convex Deep Learning due to vectorization efficiency on GPU/CPU.")

if __name__ == "__main__":
    main()