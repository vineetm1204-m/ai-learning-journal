import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------------
# 1. Synthetic Dataset Generation (Linear Regression)
# ------------------------------------------------------------
def generate_data(n_samples=1000, n_features=10, noise=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features)
    true_b = np.random.randn()
    y = X @ true_w + true_b + noise * np.random.randn(n_samples)
    # Normalize features for stable convergence comparison
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y, true_w, true_b

# ------------------------------------------------------------
# 2. Loss & Gradient Functions (MSE)
# ------------------------------------------------------------
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / 2  # 1/2 factor simplifies gradient

def compute_gradients(X_batch, y_batch, w, b):
    n = X_batch.shape[0]
    y_pred = X_batch @ w + b
    error = y_pred - y_batch
    dw = (X_batch.T @ error) / n
    db = np.mean(error)
    return dw, db

# ------------------------------------------------------------
# 3. Optimizer Implementations
# ------------------------------------------------------------
def run_gd(X, y, lr=0.1, epochs=50, batch_size=None, verbose=False):
    """
    Generic Gradient Descent Runner.
    batch_size=None  -> Batch GD (full dataset)
    batch_size=1     -> Stochastic GD (SGD)
    batch_size=k     -> Mini-batch GD
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    history = {'loss': [], 'time': [], 'epoch': []}
    
    start_time = time.time()
    
    if batch_size is None: # Batch GD
        batch_size = n_samples
    
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
            
            # Forward / Backward
            dw, db = compute_gradients(X_batch, y_batch, w, b)
            
            # Update
            w -= lr * dw
            b -= lr * db
            
            # Accumulate batch loss for monitoring
            y_pred_batch = X_batch @ w + b
            epoch_loss += compute_loss(y_batch, y_pred_batch) * len(y_batch)
        
        avg_loss = epoch_loss / n_samples
        current_time = time.time() - start_time
        
        history['loss'].append(avg_loss)
        history['time'].append(current_time)
        history['epoch'].append(epoch)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Time: {current_time:.4f}s")
            
    return w, b, history

# ------------------------------------------------------------
# 4. Experiment Configuration
# ------------------------------------------------------------
N_SAMPLES = 5000
N_FEATURES = 20
EPOCHS = 100
LR = 0.05  # Learning rate tuned for stability across all three

X, y, true_w, true_b = generate_data(N_SAMPLES, N_FEATURES)

print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features")
print(f"True Params: w_norm={np.linalg.norm(true_w):.4f}, b={true_b:.4f}\n")

# ------------------------------------------------------------
# 5. Run Experiments
# ------------------------------------------------------------
print("="*60)
print("RUNNING: BATCH GRADIENT DESCENT")
print("="*60)
w_bgd, b_bgd, hist_bgd = run_gd(X, y, lr=LR, epochs=EPOCHS, batch_size=None, verbose=True)

print("\n" + "="*60)
print("RUNNING: MINI-BATCH GRADIENT DESCENT (Batch Size=32)")
print("="*60)
w_mbgd, b_mbgd, hist_mbgd = run_gd(X, y, lr=LR, epochs=EPOCHS, batch_size=32, verbose=True)

print("\n" + "="*60)
print("RUNNING: STOCHASTIC GRADIENT DESCENT (Batch Size=1)")
print("="*60)
w_sgd, b_sgd, hist_sgd = run_gd(X, y, lr=LR, epochs=EPOCHS, batch_size=1, verbose=True)

# ------------------------------------------------------------
# 6. Analysis & Visualization
# ------------------------------------------------------------
def param_error(w_est, b_est, w_true, b_true):
    w_err = np.linalg.norm(w_est - w_true) / (np.linalg.norm(w_true) + 1e-8)
    b_err = abs(b_est - b_true) / (abs(b_true) + 1e-8)
    return w_err, b_err

print("\n" + "="*60)
print("FINAL PARAMETER ESTIMATION ERROR (Relative)")
print("="*60)
for name, w, b in [("Batch GD", w_bgd, b_bgd), 
                   ("Mini-batch GD", w_mbgd, b_mbgd), 
                   ("SGD", w_sgd, b_sgd)]:
    w_err, b_err = param_error(w, b, true_w, true_b)
    print(f"{name:15s} | Weight Err: {w_err:.4%} | Bias Err: {b_err:.4%}")

# Plotting: Loss vs Epochs & Loss vs Wall-clock Time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Loss vs Epochs
axes[0].plot(hist_bgd['epoch'], hist_bgd['loss'], label='Batch GD', linewidth=2, color='blue')
axes[0].plot(hist_mbgd['epoch'], hist_mbgd['loss'], label='Mini-batch GD (32)', linewidth=1.5, color='orange', alpha=0.8)
axes[0].plot(hist_sgd['epoch'], hist_sgd['loss'], label='SGD (1)', linewidth=1, color='green', alpha=0.6)
axes[0].set_yscale('log')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('MSE Loss (Log Scale)')
axes[0].set_title('Convergence per Epoch')
axes[0].legend()
axes[0].grid(True, which="both", ls="-", alpha=0.2)

# 2. Loss vs Wall-clock Time
axes[1].plot(hist_bgd['time'], hist_bgd['loss'], label='Batch GD', linewidth=2, color='blue')
axes[1].plot(hist_mbgd['time'], hist_mbgd['loss'], label='Mini-batch GD (32)', linewidth=1.5, color='orange', alpha=0.8)
axes[1].plot(hist_sgd['time'], hist_sgd['loss'], label='SGD (1)', linewidth=1, color='green', alpha=0.6)
axes[1].set_yscale('log')
axes[1].set_xlabel('Wall-clock Time (seconds)')
axes[1].set_ylabel('MSE Loss (Log Scale)')
axes[1].set_title('Convergence per Compute Time')
axes[1].legend()
axes[1].grid(True, which="both", ls="-", alpha=0.2)

plt.suptitle(f'Day 86: Gradient Descent Variants Comparison (LR={LR}, N={N_SAMPLES})', fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Gradient Variance Analysis (Single Epoch Snapshot)
# ------------------------------------------------------------
print("\n" + "="*60)
print("GRADIENT VARIANCE ANALYSIS (Single Epoch, Fixed Weights)")
print("="*60)
# Fix weights at a random point to compare gradient noise
w_test = np.random.randn(N_FEATURES) * 0.1
b_test = 0.1

# Batch Gradient (True Gradient)
dw_true, db_true = compute_gradients(X, y, w_test, b_test)

# Stochastic Gradients (Sample 100 individual gradients)
stoch_grads_w = []
stoch_grads_b = []
for i in range(100):
    idx = np.random.randint(0, N_SAMPLES)
    dw, db = compute_gradients(X[idx:idx+1], y[idx:idx+1], w_test, b_test)
    stoch_grads_w.append(dw)
    stoch_grads_b.append(db)

stoch_grads_w = np.array(stoch_grads_w)
stoch_grads_b = np.array(stoch_grads_b)

print(f"True Gradient Norm (w): {np.linalg.norm(dw_true):.6f}")
print(f"Mean SGD Gradient Norm (w): {np.mean(np.linalg.norm(stoch_grads_w, axis=1)):.6f}")
print(f"Std of SGD Gradient Norm (w): {np.std(np.linalg.norm(stoch_grads_w, axis=1)):.6f}")
print(f"Variance Ratio (Std/Mean): {np.std(np.linalg.norm(stoch_grads_w, axis=1)) / (np.mean(np.linalg.norm(stoch_grads_w, axis=1)) + 1e-8):.4f}")
print("\nNote: High variance in SGD gradients explains noisy convergence path.")