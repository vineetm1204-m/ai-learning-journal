import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. Synthetic Dataset Generation
# ==========================================
def generate_data(n_samples=1000, n_features=1, noise=0.1, seed=42):
    """Generate a simple linear regression problem: y = X @ w_true + b_true + noise"""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    b_true = np.random.randn()
    y = X @ w_true + b_true + noise * np.random.randn(n_samples)
    # Add bias column to X for simpler vectorized math
    X_b = np.c_[X, np.ones(n_samples)]
    w_true_full = np.r_[w_true, b_true]
    return X_b, y, w_true_full

# ==========================================
# 2. Loss & Gradient Functions (MSE)
# ==========================================
def compute_loss(X, y, w):
    """Mean Squared Error"""
    preds = X @ w
    return np.mean((preds - y) ** 2)

def compute_grad_batch(X, y, w):
    """Gradient for Full Batch GD: (2/N) * X.T @ (X @ w - y)"""
    n = X.shape[0]
    preds = X @ w
    error = preds - y
    return (2 / n) * (X.T @ error)

def compute_grad_stochastic(x_i, y_i, w):
    """Gradient for Single Sample (SGD): 2 * x_i * (x_i @ w - y_i)"""
    pred = x_i @ w
    error = pred - y_i
    return 2 * x_i * error

# ==========================================
# 3. Optimizers
# ==========================================
def run_batch_gd(X, y, w_init, lr, epochs, verbose=False):
    w = w_init.copy()
    history = []
    n = X.shape[0]
    for epoch in range(epochs):
        grad = compute_grad_batch(X, y, w)
        w -= lr * grad
        loss = compute_loss(X, y, w)
        history.append(loss)
        if verbose and (epoch % 100 == 0):
            print(f"Batch GD  Epoch {epoch:4d} | Loss: {loss:.6f}")
    return w, history

def run_sgd(X, y, w_init, lr, epochs, verbose=False):
    w = w_init.copy()
    history = []
    n = X.shape[0]
    indices = np.arange(n)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        for i in indices:
            x_i = X[i:i+1] # Keep dim (1, D)
            y_i = y[i:i+1]
            grad = compute_grad_stochastic(x_i, y_i, w)
            w -= lr * grad
            epoch_loss += compute_loss(x_i, y_i, w)
        history.append(epoch_loss / n)
        if verbose and (epoch % 100 == 0):
            print(f"SGD       Epoch {epoch:4d} | Avg Loss: {history[-1]:.6f}")
    return w, history

def run_minibatch_gd(X, y, w_init, lr, epochs, batch_size, verbose=False):
    w = w_init.copy()
    history = []
    n = X.shape[0]
    indices = np.arange(n)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            X_b = X[batch_idx]
            y_b = y[batch_idx]
            
            preds = X_b @ w
            error = preds - y_b
            grad = (2 / len(batch_idx)) * (X_b.T @ error)
            
            w -= lr * grad
            epoch_loss += compute_loss(X_b, y_b, w)
            n_batches += 1
        history.append(epoch_loss / n_batches)
        if verbose and (epoch % 100 == 0):
            print(f"Mini-Batch Epoch {epoch:4d} | Avg Loss: {history[-1]:.6f}")
    return w, history

# ==========================================
# 4. Experiment Runner
# ==========================================
def run_experiment():
    # Config
    N_SAMPLES = 5000
    N_FEATURES = 5
    EPOCHS = 200
    LR = 0.01
    BATCH_SIZE = 32
    SEED = 127

    print(f"--- Day 127: Gradient Descent Variants Comparison ---")
    print(f"Samples: {N_SAMPLES}, Features: {N_FEATURES}, LR: {LR}, Epochs: {EPOCHS}")
    print("-" * 60)

    # Data
    X, y, w_true = generate_data(N_SAMPLES, N_FEATURES, seed=SEED)
    w_init = np.zeros(X.shape[1])

    # Analytical Solution (Normal Equation) for reference
    w_opt = np.linalg.inv(X.T @ X) @ X.T @ y
    opt_loss = compute_loss(X, y, w_opt)
    print(f"Optimal Loss (Normal Eq): {opt_loss:.6f}")
    print("-" * 60)

    # 1. Batch GD
    start = time.time()
    w_batch, hist_batch = run_batch_gd(X, y, w_init, LR, EPOCHS, verbose=True)
    t_batch = time.time() - start
    print(f"Batch GD Time: {t_batch:.4f}s | Final Loss: {hist_batch[-1]:.6f} | Dist to Opt: {np.linalg.norm(w_batch - w_opt):.4f}")
    print("-" * 60)

    # 2. SGD
    # SGD usually needs lower LR or decay, but we keep LR same for comparison of dynamics
    start = time.time()
    w_sgd, hist_sgd = run_sgd(X, y, w_init, LR, EPOCHS, verbose=True)
    t_sgd = time.time() - start
    print(f"SGD Time:      {t_sgd:.4f}s | Final Loss: {hist_sgd[-1]:.6f} | Dist to Opt: {np.linalg.norm(w_sgd - w_opt):.4f}")
    print("-" * 60)

    # 3. Mini-Batch GD
    start = time.time()
    w_mini, hist_mini = run_minibatch_gd(X, y, w_init, LR, EPOCHS, BATCH_SIZE, verbose=True)
    t_mini = time.time() - start
    print(f"Mini-Batch Time: {t_mini:.4f}s | Final Loss: {hist_mini[-1]:.6f} | Dist to Opt: {np.linalg.norm(w_mini - w_opt):.4f}")
    print("-" * 60)

    # ==========================================
    # 5. Visualization
    # ==========================================
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss Curves (Log Scale)
    plt.subplot(1, 2, 1)
    plt.plot(hist_batch, label='Batch GD', linewidth=2, alpha=0.8)
    plt.plot(hist_mini, label=f'Mini-Batch (size={BATCH_SIZE})', linewidth=2, alpha=0.8)
    plt.plot(hist_sgd, label='SGD (Batch=1)', linewidth=1, alpha=0.6, color='red')
    plt.axhline(y=opt_loss, color='k', linestyle='--', label='Optimal Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 2: Trajectory in Weight Space (First 2 weights)
    plt.subplot(1, 2, 2)
    # We need to re-run storing weight history for trajectory
    # For simplicity, we just plot the final positions relative to optimum
    # But let's do a quick trajectory capture for Batch vs Mini
    def get_trajectory(optimizer_fn, **kwargs):
        w = w_init.copy()
        traj = [w.copy()]
        n = X.shape[0]
        indices = np.arange(n)
        for epoch in range(EPOCHS):
            if optimizer_fn.__name__ == 'run_sgd':
                np.random.shuffle(indices)
                for i in indices:
                    x_i = X[i:i+1]; y_i = y[i:i+1]
                    grad = compute_grad_stochastic(x_i, y_i, w)
                    w -= LR * grad
                    traj.append(w.copy())
            elif optimizer_fn.__name__ == 'run_minibatch_gd':
                np.random.shuffle(indices)
                for start_idx in range(0, n, BATCH_SIZE):
                    end_idx = min(start_idx + BATCH_SIZE, n)
                    batch_idx = indices[start_idx:end_idx]
                    X_b = X[batch_idx]; y_b = y[batch_idx]
                    preds = X_b @ w; error = preds - y_b
                    grad = (2 / len(batch_idx)) * (X_b.T @ error)
                    w -= LR * grad
                    traj.append(w.copy())
            else: # Batch
                grad = compute_grad_batch(X, y, w)
                w -= LR * grad
                traj.append(w.copy())
        return np.array(traj)

    traj_batch = get_trajectory(run_batch_gd)
    traj_mini = get_trajectory(run_minibatch_gd)
    traj_sgd = get_trajectory(run_sgd)

    # Plot first 2 weight dimensions
    plt.plot(traj_batch[:, 0], traj_batch[:, 1], 'b-o', markersize=2, label='Batch GD', alpha=0.7, linewidth=1)
    plt.plot(traj_mini[:, 0], traj_mini[:, 1], 'g-s', markersize=2, label=f'Mini-Batch', alpha=0.7, linewidth=1)
    plt.plot(traj_sgd[::50, 0], traj_sgd[::50, 1], 'r.', markersize=1, label='SGD (subsampled)', alpha=0.5) # Subsample SGD for visibility
    plt.plot(w_opt[0], w_opt[1], 'k*', markersize=15, label='Optimum')
    plt.plot(w_init[0], w_init[1], 'kx', markersize=10, label='Start')
    plt.xlabel('Weight 1')
    plt.ylabel('Weight 2')
    plt.title('Optimization Trajectory (W1 vs W2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig("day127_gd_comparison.png", dpi=150)
    print("\nPlot saved to 'day127_gd_comparison.png'")
    # plt.show() # Commented out for non-interactive environments

if __name__ == "__main__":
    run_experiment()