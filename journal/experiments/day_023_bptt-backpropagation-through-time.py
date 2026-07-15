import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 1. SIMPLE RNN FROM SCRATCH WITH MANUAL BPTT
# ============================================================

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        # For gradient checking
        self.params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        self.param_names = ['Wxh', 'Whh', 'Why', 'bh', 'by']
    
    def forward(self, inputs, h_prev):
        """
        inputs: list of (input_size, 1) arrays, length T
        h_prev: (hidden_size, 1)
        Returns: outputs list, hidden states list, cache for backward
        """
        T = len(inputs)
        hs = [h_prev.copy()]
        ys = []
        cache = []
        
        for t in range(T):
            x = inputs[t]
            h = np.tanh(self.Wxh @ x + self.Whh @ hs[-1] + self.bh)
            y = self.Why @ h + self.by
            hs.append(h)
            ys.append(y)
            cache.append((x, h, hs[-2]))  # x_t, h_t, h_{t-1}
        
        return ys, hs, cache
    
    def loss(self, ys, targets):
        """MSE loss"""
        loss = 0
        for y, target in zip(ys, targets):
            loss += 0.5 * np.sum((y - target) ** 2)
        return loss
    
    def backward(self, targets, ys, hs, cache, clip_val=5.0):
        """
        BPTT: Backpropagation Through Time
        Returns gradients for all parameters
        """
        T = len(targets)
        hidden_size = self.Whh.shape[0]
        
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # dh_next carries gradient from future time steps
        dh_next = np.zeros((hidden_size, 1))
        
        # Backward pass through time
        for t in reversed(range(T)):
            x, h, h_prev = cache[t]
            y = ys[t]
            target = targets[t]
            
            # Output gradient: dL/dy = y - target (for MSE)
            dy = y - target
            
            # Gradients for Why, by
            dWhy += dy @ h.T
            dby += dy
            
            # Gradient flowing into hidden state
            dh = self.Why.T @ dy + dh_next
            
            # Backprop through tanh: d(tanh)/dh = 1 - tanh^2
            dh_raw = (1 - h ** 2) * dh
            
            # Gradients for Wxh, Whh, bh
            dWxh += dh_raw @ x.T
            dWhh += dh_raw @ h_prev.T
            dbh += dh_raw
            
            # Gradient to propagate to previous time step
            dh_next = self.Whh.T @ dh_raw
        
        # Gradient clipping
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -clip_val, clip_val, out=grad)
        
        return [dWxh, dWhh, dWhy, dbh, dby]
    
    def update_params(self, grads, lr):
        for param, grad in zip(self.params, grads):
            param -= lr * grad
    
    def get_params_flat(self):
        return np.concatenate([p.ravel() for p in self.params])
    
    def set_params_flat(self, flat):
        idx = 0
        for i, param in enumerate(self.params):
            shape = param.shape
            size = param.size
            param[:] = flat[idx:idx+size].reshape(shape)
            idx += size


# ============================================================
# 2. GRADIENT CHECKING (NUMERICAL VS ANALYTICAL)
# ============================================================

def gradient_check(rnn, inputs, targets, h_prev, eps=1e-5):
    """Compare analytical gradients with numerical gradients"""
    ys, hs, cache = rnn.forward(inputs, h_prev)
    analytical_grads = rnn.backward(targets, ys, hs, cache, clip_val=1e9)
    
    params_flat = rnn.get_params_flat()
    numerical_grads = np.zeros_like(params_flat)
    
    for i in range(len(params_flat)):
        old_val = params_flat[i]
        
        params_flat[i] = old_val + eps
        rnn.set_params_flat(params_flat)
        ys_p, _, _ = rnn.forward(inputs, h_prev)
        loss_p = rnn.loss(ys_p, targets)
        
        params_flat[i] = old_val - eps
        rnn.set_params_flat(params_flat)
        ys_m, _, _ = rnn.forward(inputs, h_prev)
        loss_m = rnn.loss(ys_m, targets)
        
        numerical_grads[i] = (loss_p - loss_m) / (2 * eps)
        params_flat[i] = old_val
    
    rnn.set_params_flat(params_flat)
    
    # Flatten analytical grads
    analytical_flat = np.concatenate([g.ravel() for g in analytical_grads])
    
    # Relative error
    diff = np.abs(numerical_grads - analytical_flat)
    rel_error = diff / (np.abs(numerical_grads) + np.abs(analytical_flat) + 1e-10)
    
    return rel_error.max(), rel_error.mean(), analytical_flat, numerical_grads


# ============================================================
# 3. TRUNCATED BPTT DEMONSTRATION
# ============================================================

def truncated_bptt(rnn, inputs, targets, h_prev, truncate_len):
    """
    Truncated BPTT: only backpropagate through truncate_len steps
    """
    T = len(inputs)
    total_loss = 0
    all_grads = [np.zeros_like(p) for p in rnn.params]
    
    for start in range(0, T, truncate_len):
        end = min(start + truncate_len, T)
        chunk_inputs = inputs[start:end]
        chunk_targets = targets[start:end]
        
        # Forward through chunk (need hidden state from previous chunk)
        ys, hs, cache = rnn.forward(chunk_inputs, h_prev)
        loss = rnn.loss(ys, chunk_targets)
        total_loss += loss
        
        # Backward through chunk only
        grads = rnn.backward(chunk_targets, ys, hs, cache)
        for i, g in enumerate(grads):
            all_grads[i] += g
        
        # Update hidden state for next chunk (detach gradient)
        h_prev = hs[-1].copy()
    
    return total_loss, all_grads


# ============================================================
# 4. SYNTHETIC TASK: SEQUENCE ADDITION
# ============================================================

def generate_addition_task(seq_len, batch_size=1):
    """
    Input: two sequences, one with numbers, one with markers
    Target: sum of marked numbers
    """
    # Sequence 1: random numbers
    nums = np.random.uniform(0, 1, (seq_len, 1))
    # Sequence 2: two 1s at random positions, rest 0
    markers = np.zeros((seq_len, 1))
    pos1, pos2 = np.random.choice(seq_len, 2, replace=False)
    markers[pos1] = 1
    markers[pos2] = 1
    
    # Input is concatenated: [num, marker] at each step
    inputs = [np.vstack([nums[t], markers[t]]) for t in range(seq_len)]
    
    # Target is sum of marked numbers (same at every time step for simplicity)
    target_val = nums[pos1] + nums[pos2]
    targets = [np.array([[target_val]]) for _ in range(seq_len)]
    
    return inputs, targets, target_val


# ============================================================
# 5. VANISHING/EXPLODING GRADIENT VISUALIZATION
# ============================================================

def analyze_gradient_flow(rnn, seq_len=20, n_trials=50):
    """Track gradient norms through time"""
    grad_norms_per_step = []
    
    for _ in range(n_trials):
        inputs, targets, _ = generate_addition_task(seq_len)
        h_prev = np.zeros((rnn.Whh.shape[0], 1))
        ys, hs, cache = rnn.forward(inputs, h_prev)
        
        # Compute gradient at each time step individually
        T = len(targets)
        step_norms = []
        
        for t in range(T):
            # Create dummy targets: only non-zero at time t
            dummy_targets = [np.zeros_like(targets[0]) for _ in range(T)]
            dummy_targets[t] = targets[t]
            
            grads = rnn.backward(dummy_targets, ys, hs, cache, clip_val=1e9)
            # Norm of gradient w.r.t Whh at this step
            norm = np.linalg.norm(grads[1])  # dWhh
            step_norms.append(norm)
        
        grad_norms_per_step.append(step_norms)
    
    return np.array(grad_norms_per_step).mean(axis=0)


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 60)
    print("DAY 23: BPTT - BACKPROPAGATION THROUGH TIME")
    print("=" * 60)
    
    # Hyperparameters
    input_size = 2
    hidden_size = 16
    output_size = 1
    seq_len = 10
    lr = 0.01
    epochs = 200
    
    # Initialize RNN
    rnn = SimpleRNN(input_size, hidden_size, output_size)
    
    # --------------------------------------------------------
    # 1. GRADIENT CHECK
    # --------------------------------------------------------
    print("\n[1] GRADIENT CHECKING")
    print("-" * 40)
    test_inputs, test_targets, _ = generate_addition_task(5)
    h0 = np.zeros((hidden_size, 1))
    max_rel, mean_rel, _, _ = gradient_check(rnn, test_inputs, test_targets, h0)
    print(f"Max relative error: {max_rel:.2e}")
    print(f"Mean relative error: {mean_rel:.2e}")
    print("✓ Gradients correct" if max_rel < 1e-6 else "✗ Gradient mismatch!")
    
    # --------------------------------------------------------
    # 2. TRAINING WITH FULL BPTT
    # --------------------------------------------------------
    print("\n[2] TRAINING WITH FULL BPTT")
    print("-" * 40)
    losses_full = []
    
    for epoch in range(epochs):
        inputs, targets, true_sum = generate_addition_task(seq_len)
        h_prev = np.zeros((hidden_size, 1))
        
        ys, hs, cache = rnn.forward(inputs, h_prev)
        loss = rnn.loss(ys, targets)
        losses_full.append(loss)
        
        grads = rnn.backward(targets, ys, hs, cache)
        rnn.update_params(grads, lr)
        
        if (epoch + 1) % 50 == 0:
            pred = ys[-1][0, 0]
            print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, True={true_sum[0,0]:.4f}, Pred={pred:.4f}")
    
    # --------------------------------------------------------
    # 3. TRUNCATED BPTT COMPARISON
    # --------------------------------------------------------
    print("\n[3] TRUNCATED BPTT COMPARISON")
    print("-" * 40)
    
    # Fresh RNN for fair comparison
    rnn_trunc = SimpleRNN(input_size, hidden_size, output_size)
    # Copy weights from trained full BPTT
    for i in range(len(rnn.params)):
        rnn_trunc.params[i][:] = rnn.params[i]
    
    trunc_lengths = [2, 3, 5, 10]
    results = {}
    
    for trunc_len in trunc_lengths:
        rnn_t = SimpleRNN(input_size, hidden_size, output_size)
        for i in range(len(rnn.params)):
            rnn_t.params[i][:] = rnn.params[i]
        
        losses_t = []
        for epoch in range(100):
            inputs, targets, _ = generate_addition_task(seq_len)
            h_prev = np.zeros((hidden_size, 1))
            loss, grads = truncated_bptt(rnn_t, inputs, targets, h_prev, trunc_len)
            losses_t.append(loss)
            rnn_t.update_params(grads, lr)
        
        results[trunc_len] = losses_t
        print(f"Truncate={trunc_len:2d}: Final loss={losses_t[-1]:.4f}")
    
    # --------------------------------------------------------
    # 4. GRADIENT FLOW ANALYSIS
    # --------------------------------------------------------
    print("\n[4] GRADIENT FLOW ANALYSIS (Vanishing/Exploding)")
    print("-" * 40)
    
    # Test with different Whh initializations
    for scale_name, scale in [("Small (0.1)", 0.1), ("Normal (1.0)", 1.0), ("Large (2.0)", 2.0)]:
        rnn_test = SimpleRNN(input_size, hidden_size, output_size)
        rnn_test.Whh *= scale  # Scale recurrent weights
        
        grad_norms = analyze_gradient_flow(rnn_test, seq_len=15, n_trials=30)
        print(f"{scale_name:15s}: Gradient norms (t=0→14): {grad_norms[0]:.2e} → {grad_norms[-1]:.2e} "
              f"(ratio: {grad_norms[-1]/grad_norms[0]:.2e})")
    
    # --------------------------------------------------------
    # 5. VISUALIZATION
    # --------------------------------------------------------
    print("\n[5] GENERATING PLOTS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training loss
    ax = axes[0, 0]
    ax.plot(losses_full, label='Full BPTT', alpha=0.8)
    for trunc_len, losses in results.items():
        ax.plot(losses, label=f'Truncated (k={trunc_len})', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss: Full vs Truncated BPTT')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norms through time (vanishing gradient)
    ax = axes[0, 1]
    for scale_name, scale in [("Small (0.1)", 0.1), ("Normal (1.0)", 1.0), ("Large (2.0)", 2.0)]:
        rnn_test = SimpleRNN(input_size, hidden_size, output_size)
        rnn_test.Whh *= scale
        grad_norms = analyze_gradient_flow(rnn_test, seq_len=15, n_trials=30)
        ax.semilogy(range(15), grad_norms, label=scale_name, marker='o', markersize=3)
    ax.set_xlabel('Time step (t)')
    ax.set_ylabel('Avg |∂L/∂Whh| (log scale)')
    ax.set_title('Gradient Norm vs Time Step (Vanishing/Exploding)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Eigenvalues of Whh (theoretical gradient scaling)
    ax = axes[1, 0]
    for scale_name, scale in [("Small (0.1)", 0.1), ("Normal (1.0)", 1.0), ("Large (2.0)", 2.0)]:
        Whh_test = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size) * scale
        eigvals = np.linalg.eigvals(Whh_test)
        ax.scatter(eigvals.real, eigvals.imag, alpha=0.6, label=scale_name, s=20)
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')
    ax.set_aspect('equal')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title('Eigenvalues of Whh (Spectral Radius)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    
    # Plot 4: Prediction vs Target on test sequences
    ax = axes[1, 1]
    n_test = 20
    preds, trues = [], []
    for _ in range(n_test):
        inputs, targets, true_sum = generate_addition_task(seq_len)
        h_prev = np.zeros((hidden_size, 1))
        ys, _, _ = rnn.forward(inputs, h_prev)
        preds.append(ys[-1][0, 0])
        trues.append(true_sum[0, 0])
    
    ax.scatter(trues, preds, alpha=0.7)
    ax.plot([0, 2], [0, 2], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Sum')
    ax.set_ylabel('Predicted Sum')
    ax.set_title(f'Test Predictions (R² = {np.corrcoef(trues, preds)[0,1]**2:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day23_bptt_results.png', dpi=150, bbox_inches='tight')
    print("Saved: day23_bptt_results.png")
    
    # --------------------------------------------------------
    # 6. BPTT STEP-BY-STEP DEMONSTRATION
    # --------------------------------------------------------
    print("\n[6] BPTT STEP-BY-STEP (T=3)")
    print("-" * 40)
    
    demo_rnn = SimpleRNN(2, 4, 1)
    demo_inputs, demo_targets, _ = generate_addition_task(3)
    h0 = np.zeros((4, 1))
    
    print("Forward pass:")
    ys, hs, cache = demo_rnn.forward(demo_inputs, h0)
    for t in range(3):
        print(f"  t={t}: x={demo_inputs[t].ravel()}, h={hs[t+1].ravel()}, y={ys[t].ravel()}")
    
    print("\nBackward pass (BPTT):")
    grads = demo_rnn.backward(demo_targets, ys, hs, cache, clip_val=1e9)
    for name, grad in zip(demo_rnn.param_names, grads):
        print(f"  ∂L/∂{name}: shape={grad.shape}, norm={np.linalg.norm(grad):.4f}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()