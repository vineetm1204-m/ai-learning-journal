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
    b_true = np.random.randn()
    y = X @ w_true + b_true + noise * np.random.randn(n_samples)
    # Add bias column to X for vectorized implementation
    X_b = np.c_[X, np.ones(n_samples)]
    return X_b, y, np.append(w_true, b_true)

# ==========================================
# 2. LOSS & GRADIENT FUNCTIONS
# ==========================================
def mse_loss(X, y, theta):
    """Mean Squared Error Loss"""
    preds = X @ theta
    return np.mean((preds - y) ** 2)

def mse_gradient(X, y, theta):
    """Gradient of MSE: (2/N) * X.T @ (X @ theta - y)"""
    n = X.shape[0]
    preds = X @ theta
    error = preds - y
    return (2.0 / n) * (X.T @ error)

def batch_gradient(X_batch, y_batch, theta):
    """Gradient for a specific batch"""
    n = X_batch.shape[0]
    preds = X_batch @ theta
    error = preds - y_batch
    return (2.0 / n) * (X_batch.T @ error)

# ==========================================
# 3. OPTIMIZER IMPLEMENTATIONS
# ==========================================
def run_batch_gd(X, y, theta_init, lr, epochs, log_interval=10):
    """Batch Gradient Descent: Uses full dataset per step."""
    theta = theta_init.copy()
    history = {'loss': [], 'time': [], 'theta_norm': []}
    start = time.time()
    
    for epoch in range(epochs):
        grad = mse_gradient(X, y, theta)
        theta -= lr * grad
        
        if epoch % log_interval == 0:
            loss = mse_loss(X, y, theta)
            history['loss'].append(loss)
            history['time'].append(time.time() - start)
            history['theta_norm'].append(np.linalg.norm(theta))
            
    return theta, history

def run_sgd(X, y, theta_init, lr, epochs, log_interval=10):
    """Stochastic Gradient Descent: 1 sample per step."""
    theta = theta_init.copy()
    history = {'loss': [], 'time': [], 'theta_norm': []}
    n_samples = X.shape[0]
    start = time.time()
    steps_per_epoch = n_samples
    
    for epoch in range(epochs):
        # Shuffle data every epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(steps_per_epoch):
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]
            grad = batch_gradient(xi, yi, theta)
            theta -= lr * grad
            
        # Log once per epoch (after full pass)
        if epoch % log_interval == 0:
            loss = mse_loss(X, y, theta)
            history['loss'].append(loss)
            history['time'].append(time.time() - start)
            history['theta_norm'].append(np.linalg.norm(theta))
            
    return theta, history

def run_minibatch_gd(X, y, theta_init, lr, epochs, batch_size=32, log_interval=10):
    """Mini-batch Gradient Descent: batch_size samples per step."""
    theta = theta_init.copy()
    history = {'loss': [], 'time': [], 'theta_norm': []}
    n_samples = X.shape[0]
    start = time.time()
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, batch_size):
            xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]
            grad = batch_gradient(xb, yb, theta)
            theta -= lr * grad
            
        if epoch % log_interval == 0:
            loss = mse_loss(X, y, theta)
            history['loss'].append(loss)
            history['time'].append(time.time() - start)
            history['theta_norm'].append(np.linalg.norm(theta))
            
    return theta, history

# ==========================================
# 4. EXPERIMENT RUNNER
# ==========================================
def run_experiment():
    # Config
    N_SAMPLES = 5000
    N_FEATURES = 20
    EPOCHS = 200
    LR = 0.01
    BATCH_SIZE = 64
    LOG_INTERVAL = 10
    SEED = 42
    
    print(f"Generating data: {N_SAMPLES} samples, {N_FEATURES} features...")
    X, y, theta_true = generate_data(N_SAMPLES, N_FEATURES, seed=SEED)
    theta_init = np.zeros(X.shape[1])
    
    print(f"\nTrue Theta Norm: {np.linalg.norm(theta_true):.4f}")
    print(f"Initial Loss: {mse_loss(X, y, theta_init):.4f}")
    print("-" * 50)
    
    # --- Run Optimizers ---
    print("Running Batch GD...")
    theta_bgd, hist_bgd = run_batch_gd(X, y, theta_init, LR, EPOCHS, LOG_INTERVAL)
    
    print("Running Mini-batch GD...")
    theta_mbgd, hist_mbgd = run_minibatch_gd(X, y, theta_init, LR, EPOCHS, BATCH_SIZE, LOG_INTERVAL)
    
    print("Running SGD...")
    theta_sgd, hist_sgd = run_sgd(X, y, theta_init, LR, EPOCHS, LOG_INTERVAL)
    
    # --- Results Summary ---
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Method':<15} | {'Final Loss':<12} | {'Theta Error':<12} | {'Time (s)':<10} | {'Updates/Epoch'}")
    print("-"*60)
    
    def theta_error(t): return np.linalg.norm(t - theta_true)
    
    print(f"{'Batch GD':<15} | {hist_bgd['loss'][-1]:<12.6f} | {theta_error(theta_bgd):<12.6f} | {hist_bgd['time'][-1]:<10.4f} | 1")
    print(f"{'Mini-batch GD':<15} | {hist_mbgd['loss'][-1]:<12.6f} | {theta_error(theta_mbgd):<12.6f} | {hist_mbgd['time'][-1]:<10.4f} | {N_SAMPLES//BATCH_SIZE}")
    print(f"{'SGD':<15} | {hist_sgd['loss'][-1]:<12.6f} | {theta_error(theta_sgd):<12.6f} | {hist_sgd['time'][-1]:<10.4f} | {N_SAMPLES}")
    
    return hist_bgd, hist_mbgd, hist_sgd

# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_results(hist_bgd, hist_mbgd, hist_sgd):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss vs Epochs
    ax = axes[0]
    epochs_logged = np.arange(0, len(hist_bgd['loss'])) * 10 # log_interval=10
    ax.plot(epochs_logged, hist_bgd['loss'], 'b-o', label='Batch GD', markersize=4, linewidth=1.5)
    ax.plot(epochs_logged, hist_mbgd['loss'], 'g-s', label=f'Mini-batch GD (bs=64)', markersize=4, linewidth=1.5)
    ax.plot(epochs_logged, hist_sgd['loss'], 'r-^', label='SGD', markersize=4, linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss (Log Scale)')
    ax.set_title('Convergence Speed (Loss vs Epochs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs Wall-clock Time
    ax = axes[1]
    ax.plot(hist_bgd['time'], hist_bgd['loss'], 'b-o', label='Batch GD', markersize=4, linewidth=1.5)
    ax.plot(hist_mbgd['time'], hist_mbgd['loss'], 'g-s', label='Mini-batch GD', markersize=4, linewidth=1.5)
    ax.plot(hist_sgd['time'], hist_sgd['loss'], 'r-^', label='SGD', markersize=4, linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Wall-clock Time (seconds)')
    ax.set_ylabel('MSE Loss (Log Scale)')
    ax.set_title('Computational Efficiency (Loss vs Time)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Day 45: Gradient Descent Variants Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    h_bgd, h_mbgd, h_sgd = run_experiment()
    plot_results(h_bgd, h_mbgd, h_sgd)