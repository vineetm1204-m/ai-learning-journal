import numpy as np

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
np.random.seed(42)
INPUT_DIM = 1
HIDDEN_DIM = 16
OUTPUT_DIM = 1
SEQ_LENGTH = 10
LEARNING_RATE = 0.01
EPOCHS = 2000
CLIP_VALUE = 5.0  # Gradient clipping threshold

# ------------------------------------------------------------
# Data Generation: Simple Sequence Prediction (sin wave)
# ------------------------------------------------------------
def generate_data(num_samples=1000):
    X, Y = [], []
    for _ in range(num_samples):
        start = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(start, start + 4 * np.pi, SEQ_LENGTH + 1)
        seq = np.sin(t).reshape(-1, 1)
        X.append(seq[:-1])
        Y.append(seq[1:])
    return np.array(X), np.array(Y)

X_train, Y_train = generate_data(500)
X_val, Y_val = generate_data(100)

# ------------------------------------------------------------
# RNN Cell (Vanilla) - Explicit Unrolling Logic
# ------------------------------------------------------------
class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Weights
        self.Wxh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Why = np.random.randn(output_dim, hidden_dim) * 0.01
        # Biases
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
        
        # Gradients storage
        self.dWxh, self.dWhh, self.dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.dbh, self.dby = np.zeros_like(self.bh), np.zeros_like(self.by)

    def forward(self, inputs, h_prev):
        """
        Unrolls the RNN through time explicitly.
        Returns: outputs (list), hidden_states (list including h_prev at index 0)
        """
        T = len(inputs)
        h = h_prev.copy()
        hidden_states = [h] # h_0 at index 0
        outputs = []
        
        for t in range(T):
            x_t = inputs[t].reshape(-1, 1)
            # h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
            h = np.tanh(self.Wxh @ x_t + self.Whh @ h + self.bh)
            hidden_states.append(h)
            # y_t = Why * h_t + by
            y_t = self.Why @ h + self.by
            outputs.append(y_t)
            
        return outputs, hidden_states

    def backward(self, inputs, targets, hidden_states, outputs):
        """
        Backpropagation Through Time (BPTT).
        Unrolls the computational graph backwards.
        """
        T = len(inputs)
        # Initialize gradient accumulators
        self.dWxh.fill(0); self.dWhh.fill(0); self.dWhy.fill(0)
        self.dbh.fill(0); self.dby.fill(0)
        
        # dh_next represents gradient flowing from future time step (t+1)
        dh_next = np.zeros_like(hidden_states[0])
        loss = 0.0
        
        for t in reversed(range(T)):
            x_t = inputs[t].reshape(-1, 1)
            y_t = outputs[t]
            target_t = targets[t].reshape(-1, 1)
            h_t = hidden_states[t+1]
            h_prev = hidden_states[t]
            
            # 1. Output Layer Gradient (MSE Loss: 0.5 * (y - target)^2)
            dy = (y_t - target_t) # dLoss/dy
            loss += 0.5 * np.sum(dy ** 2)
            
            # Gradients for Why, by
            self.dWhy += dy @ h_t.T
            self.dby += dy
            
            # 2. Hidden State Gradient
            # dh = (Why^T @ dy) + dh_next (from future)
            dh = self.Why.T @ dy + dh_next
            
            # 3. Tanh Non-linearity Gradient
            # d(tanh(z))/dz = 1 - tanh^2(z) = 1 - h^2
            dtanh = (1 - h_t ** 2) * dh
            
            # 4. Parameter Gradients (Accumulate)
            self.dbh += dtanh
            self.dWxh += dtanh @ x_t.T
            self.dWhh += dtanh @ h_prev.T
            
            # 5. Pass gradient to previous time step
            dh_next = self.Whh.T @ dtanh
            
        # Gradient Clipping (Global Norm)
        for grad in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
            np.clip(grad, -CLIP_VALUE, CLIP_VALUE, out=grad)
            
        return loss / T

    def update_params(self, lr):
        for param, grad in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]
        ):
            param -= lr * grad

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
def train():
    model = VanillaRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    h_init = np.zeros((HIDDEN_DIM, 1))
    
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12}")
    print("-" * 38)
    
    for epoch in range(1, EPOCHS + 1):
        # --- Training Step (Batch Size = 1 for simplicity of unrolling demo) ---
        idx = np.random.randint(0, len(X_train))
        x_seq = X_train[idx] # (T, 1)
        y_seq = Y_train[idx]
        
        # Forward (Unroll Forward)
        outputs, hidden_states = model.forward(x_seq, h_init)
        
        # Backward (Unroll Backward / BPTT)
        loss = model.backward(x_seq, y_seq, hidden_states, outputs)
        
        # Update
        model.update_params(LEARNING_RATE)
        
        # --- Validation ---
        if epoch % 200 == 0 or epoch == 1:
            val_losses = []
            for i in range(len(X_val)):
                v_out, v_hid = model.forward(X_val[i], h_init)
                # Re-use backward for loss calc without updating grads (hacky but ok for demo)
                # Better: separate eval forward pass
                v_loss = 0.5 * np.mean((np.array(v_out).squeeze() - Y_val[i].squeeze())**2)
                val_losses.append(v_loss)
            avg_val_loss = np.mean(val_losses)
            print(f"{epoch:6d} | {loss:12.6f} | {avg_val_loss:12.6f}")

    return model

# ------------------------------------------------------------
# Qualitative Evaluation: Generate Sequence
# ------------------------------------------------------------
def generate_sequence(model, seed_seq, steps=20):
    h = np.zeros((HIDDEN_DIM, 1))
    # Prime with seed
    for x in seed_seq:
        _, h = model.forward([x.reshape(-1,1)], h)
    
    generated = []
    x_next = seed_seq[-1].reshape(-1, 1)
    for _ in range(steps):
        out, h = model.forward([x_next], h)
        x_next = out[0] # Feed prediction as next input (autoregressive)
        generated.append(x_next.item())
    return generated

# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    print("--- Day 63: RNN Unrolled Through Time (NumPy From Scratch) ---")
    trained_model = train()
    
    print("\n--- Autoregressive Generation Test ---")
    # Seed with first 5 steps of a validation sequence
    seed = X_val[0][:5]
    preds = generate_sequence(trained_model, seed, steps=15)
    targets = Y_val[0][5:20].squeeze()
    
    print(f"{'Step':>4} | {'Target':>10} | {'Predicted':>10} | {'Diff':>10}")
    print("-" * 42)
    for i, (t, p) in enumerate(zip(targets, preds)):
        print(f"{i+1:4d} | {t:10.4f} | {p:10.4f} | {abs(t-p):10.4f}")