import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. MANUAL RNN + BPTT FROM SCRATCH (NumPy)
# ============================================================
class ManualRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        # Xavier initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        """inputs: list of (input_size, 1) arrays, length T"""
        T = len(inputs)
        hs = {}
        hs[-1] = h_prev.copy()
        ys = {}
        for t in range(T):
            hs[t] = np.tanh(self.Wxh @ inputs[t] + self.Whh @ hs[t-1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by
        return ys, hs

    def loss(self, ys, targets):
        """MSE loss"""
        T = len(ys)
        loss = 0.0
        for t in range(T):
            loss += 0.5 * np.sum((ys[t] - targets[t]) ** 2)
        return loss

    def backward(self, inputs, hs, ys, targets, h_prev):
        """BPTT: compute gradients wrt all parameters"""
        T = len(inputs)
        # Gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(T)):
            dy = ys[t] - targets[t]  # dL/dy (MSE)
            dWhy += dy @ hs[t].T
            dby += dy

            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh  # tanh derivative

            dbh += dh_raw
            dWxh += dh_raw @ inputs[t].T
            dWhh += dh_raw @ hs[t-1].T
            dh_next = self.Whh.T @ dh_raw

        # Gradient clipping (essential for BPTT)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        grads = {'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy, 'bh': dbh, 'by': dby}
        return grads, dh_next

    def update(self, grads, lr):
        for key in grads:
            getattr(self, key) -= lr * grads[key]


# ============================================================
# 2. SYNTHETIC TASK: SEQUENCE ADDITION (T=10)
# ============================================================
def generate_addition_task(batch_size, seq_len=10):
    """Two numbers encoded in sequence, target = sum"""
    X = np.zeros((batch_size, seq_len, 2))
    y = np.zeros((batch_size, 1))
    for i in range(batch_size):
        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-1, 1)
        # Encode at random positions
        pos_a = np.random.randint(0, seq_len//2)
        pos_b = np.random.randint(seq_len//2, seq_len)
        X[i, pos_a, 0] = a
        X[i, pos_b, 1] = b
        y[i, 0] = a + b
    return X, y


# ============================================================
# 3. PYTORCH RNN WITH GRADIENT TRACKING
# ============================================================
class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1, :])  # Use last hidden state
        return out, h


def track_gradient_norms(model, seq_len=20):
    """Track gradient norms wrt hidden states at each time step"""
    model.train()
    x = torch.randn(1, seq_len, 2)
    target = torch.randn(1, 1)
    
    # Forward with retain_graph for intermediate gradients
    h = torch.zeros(1, 1, model.hidden_size, requires_grad=True)
    hiddens = []
    x.requires_grad_(True)
    
    # Manual unroll to capture each time step's hidden state
    h_t = h
    for t in range(seq_len):
        x_t = x[:, t:t+1, :]
        h_t = torch.tanh(model.rnn.weight_ih_l0 @ x_t.transpose(1,2) + 
                         model.rnn.weight_hh_l0 @ h_t.transpose(1,2) + 
                         model.rnn.bias_ih_l0.unsqueeze(1) + 
                         model.rnn.bias_hh_l0.unsqueeze(1))
        h_t = h_t.transpose(1,2)
        hiddens.append(h_t)
    
    h_final = hiddens[-1]
    out = model.fc(h_final.squeeze(0))
    loss = nn.MSELoss()(out, target)
    
    # Compute gradients wrt each hidden state
    grad_norms = []
    for h_t in hiddens:
        grad = torch.autograd.grad(loss, h_t, retain_graph=True, allow_unused=True)[0]
        if grad is not None:
            grad_norms.append(grad.norm().item())
        else:
            grad_norms.append(0.0)
    return grad_norms


# ============================================================
# 4. TRUNCATED BPTT DEMONSTRATION
# ============================================================
def train_truncated_bptt(model, data_loader, truncate_len, epochs=5, lr=0.01):
    """Train with truncated BPTT"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            seq_len = X_batch.size(1)
            h = None
            total_loss = 0
            
            for t in range(0, seq_len, truncate_len):
                end = min(t + truncate_len, seq_len)
                x_chunk = X_batch[:, t:end, :]
                y_chunk = y_batch if end == seq_len else None
                
                out, h = model(x_chunk, h)
                h = h.detach()  # TRUNCATE: stop gradient flow
                
                if y_chunk is not None:
                    loss = criterion(out, y_chunk)
                    total_loss += loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += total_loss.item()
        losses.append(epoch_loss / len(data_loader))
    return losses


def train_full_bptt(model, data_loader, epochs=5, lr=0.01):
    """Train with full BPTT (no truncation)"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            out, _ = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(data_loader))
    return losses


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================
def run_experiment():
    print("=" * 60)
    print("DAY 64: BPTT - BACKPROPAGATION THROUGH TIME")
    print("=" * 60)
    
    # --- Config ---
    INPUT_SIZE = 2
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 1
    SEQ_LEN = 15
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.01
    
    # --- Data ---
    X_np, y_np = generate_addition_task(1000, SEQ_LEN)
    X_tensor = torch.FloatTensor(X_np)
    y_tensor = torch.FloatTensor(y_np)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- 1. Manual RNN + BPTT (NumPy) ---
    print("\n[1] Manual RNN + BPTT (NumPy) on Addition Task")
    print("-" * 50)
    rnn = ManualRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    h_prev = np.zeros((HIDDEN_SIZE, 1))
    manual_losses = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i in range(0, len(X_np), BATCH_SIZE):
            X_batch = X_np[i:i+BATCH_SIZE]
            y_batch = y_np[i:i+BATCH_SIZE]
            batch_loss = 0
            for j in range(len(X_batch)):
                inputs = [X_batch[j, t:t+1].T for t in range(SEQ_LEN)]
                targets = [np.array([[y_batch[j, 0]]]) for _ in range(SEQ_LEN)]
                ys, hs = rnn.forward(inputs, h_prev)
                loss = rnn.loss(ys, targets)
                grads, _ = rnn.backward(inputs, hs, ys, targets, h_prev)
                rnn.update(grads, LR)
                batch_loss += loss
            epoch_loss += batch_loss / len(X_batch)
        manual_losses.append(epoch_loss / (len(X_np) // BATCH_SIZE))
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Loss = {manual_losses[-1]:.6f}")
    
    # --- 2. PyTorch Full BPTT ---
    print("\n[2] PyTorch Full BPTT")
    print("-" * 50)
    model_full = TorchRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    full_losses = train_full_bptt(model_full, loader, epochs=EPOCHS, lr=LR)
    for epoch, loss in enumerate(full_losses):
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Loss = {loss:.6f}")
    
    # --- 3. PyTorch Truncated BPTT ---
    print("\n[3] PyTorch Truncated BPTT (truncate=5)")
    print("-" * 50)
    model_trunc = TorchRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    trunc_losses = train_truncated_bptt(model_trunc, loader, truncate_len=5, epochs=EPOCHS, lr=LR)
    for epoch, loss in enumerate(trunc_losses):
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Loss = {loss:.6f}")
    
    # --- 4. Gradient Norm Analysis ---
    print("\n[4] Gradient Norm Decay Across Time Steps")
    print("-" * 50)
    model_grad = TorchRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    grad_norms = track_gradient_norms(model_grad, seq_len=20)
    print("  Time step : Gradient Norm")
    for t, gn in enumerate(grad_norms):
        print(f"    t={t:2d}      : {gn:.6f}")
    
    # --- 5. Vanishing/Exploding Gradient Demo ---
    print("\n[5] Vanishing/Exploding Gradient Demonstration")
    print("-" * 50)
    # Simulate gradient flow through Whh^T repeatedly
    Whh = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1  # Small init -> vanishing
    # Whh = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 1.5  # Large init -> exploding
    eigvals = np.linalg.eigvals(Whh)
    spectral_radius = np.max(np.abs(eigvals))
    print(f"  Whh spectral radius: {spectral_radius:.4f}")
    print(f"  -> {'Vanishing' if spectral_radius < 1 else 'Exploding'} gradients expected")
    
    # Track norm of (Whh^T)^k
    powers = []
    M = Whh.T.copy()
    for k in range(1, 21):
        powers.append(np.linalg.norm(M, 2))
        M = M @ Whh.T
    print("  ||(Whh^T)^k||_2 for k=1..20:")
    for k, p in enumerate(powers):
        print(f"    k={k+1:2d}: {p:.6f}")
    
    # --- 6. Visualization ---
    print("\n[6] Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(manual_losses, label='Manual BPTT (NumPy)', marker='o')
    ax.plot(full_losses, label='PyTorch Full BPTT', marker='s')
    ax.plot(trunc_losses, label='PyTorch Truncated BPTT (k=5)', marker='^')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Gradient norms across time
    ax = axes[0, 1]
    ax.plot(range(1, len(grad_norms)+1), grad_norms, marker='o', color='red')
    ax.set_xlabel('Time Step (t)')
    ax.set_ylabel('||∂L/∂h_t||')
    ax.set_title('Gradient Norm Decay Across Time (Full BPTT)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Gradient norm powers
    ax = axes[1, 0]
    ax.plot(range(1, 21), powers, marker='s', color='green')
    ax.set_xlabel('Power k')
    ax.set_ylabel('||(Whh^T)^k||_2')
    ax.set_title('Gradient Amplification/Attenuation Over k Steps')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Truncation effect schematic
    ax = axes[1, 1]
    T = 15
    trunc = 5
    for t in range(T):
        color = 'blue' if t % trunc == 0 else 'lightblue'
        ax.bar(t, 1, bottom=0, color=color, edgecolor='black', width=0.8)
        if t % trunc == 0 and t > 0:
            ax.annotate('✂', xy=(t-0.2, 1.05), fontsize=14, color='red')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Gradient Flow')
    ax.set_title(f'Truncated BPTT (truncate={trunc}): Gradient Blocked at Cuts')
    ax.set_ylim(0, 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('day64_bptt_experiment.png', dpi=150, bbox_inches='tight')
    print("  Saved: day64_bptt_experiment.png")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. BPTT unrolls the RNN through time and applies chain rule.")
    print("2. Gradients flow backward: ∂L/∂h_t = (∂L/∂h_{t+1}) * (∂h_{t+1}/∂h_t) + ∂L/∂h_t|direct")
    print("3. ∂h_{t+1}/∂h_t = Whh^T * diag(1 - h_t^2) -> repeated multiplication causes vanishing/exploding.")
    print("4. Gradient clipping (norm <= 5) is essential for stability.")
    print("5. Truncated BPTT (TBPTT) cuts gradient flow every k steps -> faster, less memory, but biased gradients.")
    print("6. Spectral radius of Whh determines gradient dynamics: <1 vanishes, >1 explodes.")
    print("7. LSTMs/GRUs mitigate this via gating (additive gradient paths).")


if __name__ == "__main__":
    run_experiment()