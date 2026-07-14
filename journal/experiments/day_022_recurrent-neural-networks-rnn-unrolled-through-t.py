import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------
# 1. Synthetic sequence task: predict next value in sine wave
# ------------------------------------------------------------
SEQ_LEN = 20
N_SAMPLES = 1000
T = np.linspace(0, 4 * np.pi, SEQ_LEN + 1)
DATA = np.sin(T).astype(np.float32)

X = torch.tensor(DATA[:-1]).unsqueeze(1).unsqueeze(0).repeat(N_SAMPLES, 1, 1)
y = torch.tensor(DATA[1:]).unsqueeze(1).unsqueeze(0).repeat(N_SAMPLES, 1, 1)

# Add small noise
X += 0.02 * torch.randn_like(X)
y += 0.02 * torch.randn_like(y)

train_split = int(0.8 * N_SAMPLES)
X_train, X_val = X[:train_split], X[train_split:]
y_train, y_val = y[:train_split], y[train_split:]

# ------------------------------------------------------------
# 2. Manual RNN cell (shows unrolling explicitly)
# ------------------------------------------------------------
class ManualRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Parameter(torch.randn(1, hidden_size) * 0.1)
        self.b_y = nn.Parameter(torch.zeros(1))

    def forward(self, x, h_prev):
        # x: (batch, input_size), h_prev: (batch, hidden_size)
        h = torch.tanh(x @ self.W_xh.t() + h_prev @ self.W_hh.t() + self.b_h)
        y = h @ self.W_hy.t() + self.b_y
        return y, h

# ------------------------------------------------------------
# 3. Unrolled RNN module (explicit time steps)
# ------------------------------------------------------------
class UnrolledRNN(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super().__init__()
        self.cell = ManualRNNCell(input_size, hidden_size)
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def forward(self, x, h0=None):
        # x: (batch, seq_len, input_size)
        batch = x.size(0)
        if h0 is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device)
        else:
            h = h0

        outputs = []
        hidden_states = [h]  # store for visualization

        for t in range(self.seq_len):
            x_t = x[:, t, :]           # (batch, input_size)
            y_t, h = self.cell(x_t, h) # unrolled step
            outputs.append(y_t)
            hidden_states.append(h)

        return torch.stack(outputs, dim=1), hidden_states

# ------------------------------------------------------------
# 4. Training loop
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnrolledRNN(input_size=1, hidden_size=32, seq_len=SEQ_LEN).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

train_losses, val_losses = [], []

print("Training unrolled RNN...")
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    preds, _ = model(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_preds, _ = model(X_val)
        val_loss = criterion(val_preds, y_val)
        val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d} | Train: {loss.item():.6f} | Val: {val_loss.item():.6f}")

# ------------------------------------------------------------
# 5. Visualization: unrolled computation graph + predictions
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_seq = X_val[:1]  # (1, seq_len, 1)
    preds, hidden_states = model(test_seq)
    preds = preds.cpu().numpy().squeeze()
    target = y_val[0].cpu().numpy().squeeze()
    input_seq = test_seq.cpu().numpy().squeeze()

    # Hidden state evolution (first 3 dims)
    h_evolution = torch.stack(hidden_states, dim=0).cpu().numpy()  # (seq_len+1, 1, hidden)
    h_evolution = h_evolution.squeeze(1)[:, :3]  # (seq_len+1, 3)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 5a. Loss curves
axes[0, 0].plot(train_losses, label="Train")
axes[0, 0].plot(val_losses, label="Val")
axes[0, 0].set_yscale("log")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].set_title("Training Curves")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 5b. Sequence prediction
t = np.arange(SEQ_LEN)
axes[0, 1].plot(t, input_seq, "o-", label="Input (sin(t))", alpha=0.7)
axes[0, 1].plot(t, target, "s-", label="Target (sin(t+dt))", alpha=0.7)
axes[0, 1].plot(t, preds, "^-", label="Predicted", alpha=0.9)
axes[0, 1].set_xlabel("Time step")
axes[0, 1].set_ylabel("Value")
axes[0, 1].set_title("Unrolled RNN: One-Step-Ahead Prediction")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 5c. Hidden state trajectories (unrolled through time)
for i in range(3):
    axes[1, 0].plot(range(SEQ_LEN + 1), h_evolution[:, i], label=f"h[{i}]")
axes[1, 0].set_xlabel("Time step (unrolled)")
axes[1, 0].set_ylabel("Activation")
axes[1, 0].set_title("Hidden State Evolution Through Time")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5d. Computation graph schematic (text)
axes[1, 1].axis("off")
graph_text = """
UNROLLED RNN COMPUTATION GRAPH (seq_len=5)

t=0:  x₀ → [RNN Cell] → h₀ → y₀
       ↑              ↓
t=1:  x₁ → [RNN Cell] → h₁ → y₁
       ↑              ↓
t=2:  x₂ → [RNN Cell] → h₂ → y₂
       ↑              ↓
t=3:  x₃ → [RNN Cell] → h₃ → y₃
       ↑              ↓
t=4:  x₄ → [RNN Cell] → h₄ → y₄

Shared weights across time:
  W_xh, W_hh, b_h, W_hy, b_y

Backprop Through Time (BPTT):
  ∂L/∂W = Σ_t ∂L/∂y_t · ∂y_t/∂h_t · (∏_{k=t}^1 ∂h_k/∂h_{k-1}) · ∂h_0/∂W
"""
axes[1, 1].text(0.05, 0.95, graph_text, transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.savefig("day22_rnn_unrolled.png", dpi=150)
print("\nSaved visualization to day22_rnn_unrolled.png")

# ------------------------------------------------------------
# 6. Gradient flow analysis (vanishing/exploding demo)
# ------------------------------------------------------------
print("\n--- Gradient Flow Analysis ---")
model.train()
optimizer.zero_grad()
sample_x = X_train[:16]
sample_y = y_train[:16]
preds, hidden_states = model(sample_x)
loss = criterion(preds, sample_y)
loss.backward()

# Check gradient norms per parameter
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        param_norm = param.data.norm().item()
        print(f"{name:12s} | grad_norm: {grad_norm:.4f} | param_norm: {param_norm:.4f} | ratio: {grad_norm/param_norm:.4f}")

# Hidden state gradient flow (approximate)
h_grads = []
for h in hidden_states[1:]:  # skip h0
    if h.grad is not None:
        h_grads.append(h.grad.norm(dim=1).mean().item())
    else:
        h_grads.append(0.0)

print(f"\nHidden state gradient norms per time step: {[f'{g:.4f}' for g in h_grads]}")
print("(Decaying gradients → vanishing gradient problem)")

print("\n✓ Day 22 complete: RNN unrolled through time demonstrated.")