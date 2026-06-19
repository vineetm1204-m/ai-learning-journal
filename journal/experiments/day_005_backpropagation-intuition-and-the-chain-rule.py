import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Day 5: Backpropagation Intuition & The Chain Rule
# A self-contained micro-experiment implementing a 2-layer
# neural network from scratch (NumPy only) to visualize
# gradient flow and verify the chain rule numerically.
# ============================================================

# ------------------------------------------------------------
# 1. Configuration & Reproducibility
# ------------------------------------------------------------
np.random.seed(42)
N_SAMPLES = 200
INPUT_DIM = 2
HIDDEN_DIM = 8
OUTPUT_DIM = 1
LR = 0.1
EPOCHS = 500
PRINT_EVERY = 100

# ------------------------------------------------------------
# 2. Synthetic Dataset: Non-linear separation (XOR-like)
# ------------------------------------------------------------
def make_moons(n_samples=200, noise=0.15):
    """Generate two interleaving half circles."""
    n = n_samples // 2
    outer_circ_x = np.cos(np.linspace(0, np.pi, n))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n)) - 0.5

    X = np.vstack([
        np.column_stack([outer_circ_x, outer_circ_y]),
        np.column_stack([inner_circ_x, inner_circ_y])
    ])
    y = np.hstack([np.zeros(n), np.ones(n)]).reshape(-1, 1)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx] + np.random.randn(n_samples, 2) * noise, y[idx]

X, y = make_moons(N_SAMPLES)

# ------------------------------------------------------------
# 3. Activation Functions & Derivatives (The "Local Gradients")
# ------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def bce_loss(y_true, y_pred):
    # Binary Cross Entropy
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_grad(y_true, y_pred):
    # dL/dy_pred
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])

# ------------------------------------------------------------
# 4. Network Initialization
# ------------------------------------------------------------
W1 = np.random.randn(INPUT_DIM, HIDDEN_DIM) * np.sqrt(2. / INPUT_DIM) # He init
b1 = np.zeros((1, HIDDEN_DIM))
W2 = np.random.randn(HIDDEN_DIM, OUTPUT_DIM) * np.sqrt(2. / HIDDEN_DIM)
b2 = np.zeros((1, OUTPUT_DIM))

params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# ------------------------------------------------------------
# 5. Forward Pass (Computational Graph Construction)
# ------------------------------------------------------------
def forward(X, params):
    # Layer 1: Linear -> ReLU
    Z1 = X @ params['W1'] + params['b1']
    A1 = relu(Z1)
    
    # Layer 2: Linear -> Sigmoid
    Z2 = A1 @ params['W2'] + params['b2']
    A2 = sigmoid(Z2)
    
    cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

# ------------------------------------------------------------
# 6. Backward Pass (Chain Rule Implementation)
# ------------------------------------------------------------
def backward(y_true, cache, params):
    m = y_true.shape[0]
    grads = {}
    
    # --- Output Layer Gradients ---
    # dL/dZ2 = dL/dA2 * dA2/dZ2
    # dL/dA2 from BCE: (A2 - Y) / (A2*(1-A2)*m) ... simplified for Sigmoid+BCE:
    # dL/dZ2 = (A2 - Y) / m
    dZ2 = (cache['A2'] - y_true) / m
    
    grads['W2'] = cache['A1'].T @ dZ2
    grads['b2'] = np.sum(dZ2, axis=0, keepdims=True)
    
    # --- Hidden Layer Gradients (Chain Rule Propagation) ---
    # dL/dA1 = dL/dZ2 * dZ2/dA1 = dZ2 @ W2.T
    dA1 = dZ2 @ params['W2'].T
    
    # dL/dZ1 = dL/dA1 * dA1/dZ1 (ReLU grad)
    dZ1 = dA1 * relu_grad(cache['Z1'])
    
    grads['W1'] = cache['X'].T @ dZ1
    grads['b1'] = np.sum(dZ1, axis=0, keepdims=True)
    
    return grads

# ------------------------------------------------------------
# 7. Numerical Gradient Checker (Sanity Check for Chain Rule)
# ------------------------------------------------------------
def numerical_gradient_check(X, y, params, eps=1e-5):
    print("\n--- Numerical Gradient Verification ---")
    _, cache = forward(X, params)
    analytical_grads = backward(y, cache, params)
    
    max_rel_error = 0
    for key in params:
        param = params[key]
        grad_analytical = analytical_grads[key]
        grad_numerical = np.zeros_like(param)
        
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_val = param[ix]
            
            param[ix] = old_val + eps
            loss_plus, _ = forward(X, params)
            loss_plus = bce_loss(y, loss_plus)
            
            param[ix] = old_val - eps
            loss_minus, _ = forward(X, params)
            loss_minus = bce_loss(y, loss_minus)
            
            grad_numerical[ix] = (loss_plus - loss_minus) / (2 * eps)
            param[ix] = old_val # restore
            
            it.iternext()
        
        # Relative error
        diff = np.abs(grad_numerical - grad_analytical)
        denom = np.maximum(1e-8, np.abs(grad_numerical) + np.abs(grad_analytical))
        rel_error = np.max(diff / denom)
        print(f"{key}: Max Relative Error = {rel_error:.2e}")
        max_rel_error = max(max_rel_error, rel_error)
    
    assert max_rel_error < 1e-7, "Gradient Check Failed!"
    print("Gradient Check PASSED. Chain Rule implementation is correct.\n")

# ------------------------------------------------------------
# 8. Training Loop with Gradient Norm Logging
# ------------------------------------------------------------
def train(X, y, params, epochs, lr):
    history = {'loss': [], 'grad_norm_W1': [], 'grad_norm_W2': []}
    
    for epoch in range(epochs):
        # Forward
        y_pred, cache = forward(X, params)
        loss = bce_loss(y, y_pred)
        
        # Backward
        grads = backward(y, cache, params)
        
        # Update (SGD)
        for key in params:
            params[key] -= lr * grads[key]
        
        # Logging
        history['loss'].append(loss)
        history['grad_norm_W1'].append(np.linalg.norm(grads['W1']))
        history['grad_norm_W2'].append(np.linalg.norm(grads['W2']))
        
        if (epoch + 1) % PRINT_EVERY == 0:
            acc = np.mean((y_pred > 0.5) == y)
            print(f"Epoch {epoch+1:4d} | Loss: {loss:.4f} | Acc: {acc:.4f} | "
                  f"||dW1||: {history['grad_norm_W1'][-1]:.4f} | "
                  f"||dW2||: {history['grad_norm_W2'][-1]:.4f}")
            
    return history

# ------------------------------------------------------------
# 9. Visualization: Decision Boundary & Gradient Flow
# ------------------------------------------------------------
def plot_results(X, y, params, history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Decision Boundary
    ax = axes[0]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds, _ = forward(grid, params)
    preds = preds.reshape(xx.shape)
    
    ax.contourf(xx, yy, preds, levels=50, cmap='RdBu', alpha=0.6)
    ax.contour(xx, yy, preds, levels=[0.5], colors='k', linewidths=2)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdBu', edgecolors='k', s=40)
    ax.set_title("Learned Decision Boundary (Epoch 500)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    plt.colorbar(scatter, ax=ax)
    
    # 2. Loss Curve
    ax = axes[1]
    ax.plot(history['loss'], label='BCE Loss', color='tab:blue')
    ax.set_yscale('log')
    ax.set_title("Training Loss (Log Scale)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3); ax.legend()
    
    # 3. Gradient Norms (Vanishing/Exploding Check)
    ax = axes[2]
    ax.plot(history['grad_norm_W1'], label='||dW1|| (Input->Hidden)', color='tab:orange')
    ax.plot(history['grad_norm_W2'], label='||dW2|| (Hidden->Output)', color='tab:green')
    ax.set_title("Gradient Norms per Layer (Chain Rule Flow)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("L2 Norm")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3); ax.legend()
    
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 10. Main Execution Block
# ------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("DAY 5: BACKPROPAGATION INTUITION & CHAIN RULE")
    print("="*60)
    print(f"Architecture: {INPUT_DIM} -> ReLU({HIDDEN_DIM}) -> Sigmoid({OUTPUT_DIM})")
    print(f"Samples: {N_SAMPLES} | LR: {LR} | Epochs: {EPOCHS}")
    
    # 1. Verify Math before training
    numerical_gradient_check(X, y, params)
    
    # 2. Train
    print("--- Starting Training ---")
    history = train(X, y, params, EPOCHS, LR)
    
    # 3. Final Evaluation
    y_pred_final, _ = forward(X, params)
    final_acc = np.mean((y_pred_final > 0.5) == y)
    final_loss = bce_loss(y, y_pred_final)
    print(f"\nFinal Loss: {final_loss:.4f} | Final Accuracy: {final_acc:.4f}")
    
    # 4. Visualize
    plot_results(X, y, params, history)