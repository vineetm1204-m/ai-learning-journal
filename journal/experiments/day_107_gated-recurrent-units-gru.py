import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# Configuration
# ==========================================
SEQ_LEN = 20
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
OUTPUT_SIZE = 1
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 107
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# Synthetic Data: Sine Wave Prediction
# ==========================================
def generate_data(num_samples=1000):
    """Generate sequences of sine waves with random phase shifts."""
    t = np.linspace(0, 4 * np.pi, SEQ_LEN + 1)
    base_wave = np.sin(t)
    
    X, y = [], []
    for _ in range(num_samples):
        phase = np.random.uniform(0, 2 * np.pi)
        wave = np.sin(t + phase)
        # Add slight noise
        wave += np.random.normal(0, 0.05, size=wave.shape)
        
        # Input: first SEQ_LEN steps, Target: last SEQ_LEN steps (shifted by 1)
        X.append(wave[:-1])
        y.append(wave[1:])
        
    X = np.array(X).reshape(-1, SEQ_LEN, INPUT_SIZE).astype(np.float32)
    y = np.array(y).reshape(-1, SEQ_LEN, OUTPUT_SIZE).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

X_train, y_train = generate_data(800)
X_val, y_val = generate_data(200)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# GRU Model (Using nn.GRU for production style)
# ==========================================
class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch_first=True expects (batch, seq, feature)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        # x shape: (batch, seq_len, input_size)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
        out, hn = self.gru(x, h0)
        # out shape: (batch, seq_len, hidden_size)
        out = self.fc(out)
        return out, hn

model = GRUPredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================
# Training Loop
# ==========================================
print(f"Day 107: GRU Experiment | Device: {DEVICE}")
print(f"Model Params: {sum(p.numel() for p in model.parameters())}")

history = {'train_loss': [], 'val_loss': []}

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        optimizer.zero_grad()
        preds, _ = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        # Gradient clipping is crucial for RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
    
    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())
            
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    history['train_loss'].append(avg_train)
    history['val_loss'].append(avg_val)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

# ==========================================
# Qualitative Evaluation: Autoregressive Generation
# ==========================================
model.eval()
with torch.no_grad():
    # Pick one validation sequence seed (first 5 steps)
    seed_idx = 0
    seed_seq = X_val[seed_idx:seed_idx+1].to(DEVICE) # (1, 20, 1)
    true_future = y_val[seed_idx:seed_idx+1].to(DEVICE) # (1, 20, 1)
    
    # Teacher Forcing: Feed full sequence
    tf_preds, _ = model(seed_seq)
    
    # Autoregressive: Feed 1 step, predict next, feed prediction...
    ar_steps = 40
    ar_input = seed_seq[:, -1:, :] # Start with last known step (1, 1, 1)
    h_state = None
    ar_preds = []
    
    for _ in range(ar_steps):
        pred, h_state = model(ar_input, h_state)
        ar_preds.append(pred.cpu().numpy())
        ar_input = pred # Feed prediction back in
        
    ar_preds = np.concatenate(ar_preds, axis=1).squeeze() # (40,)

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(12, 8))

# 1. Loss Curves
plt.subplot(2, 1, 1)
plt.plot(history['train_loss'], label='Train MSE', alpha=0.8)
plt.plot(history['val_loss'], label='Val MSE', alpha=0.8)
plt.yscale('log')
plt.title("Day 107: GRU Training Dynamics (Sine Wave Prediction)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Log Scale)")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Predictions vs Ground Truth
plt.subplot(2, 1, 2)
t_hist = np.arange(SEQ_LEN)
t_tf = np.arange(SEQ_LEN)
t_ar = np.arange(SEQ_LEN, SEQ_LEN + ar_steps)

# Ground Truth History + Future
gt_full = torch.cat([X_val[seed_idx], y_val[seed_idx]], dim=0).squeeze().numpy()
plt.plot(np.arange(SEQ_LEN + 1), gt_full, 'k-', label='Ground Truth', linewidth=2, alpha=0.7)

# Teacher Forcing Prediction (One-step ahead)
tf_plot = tf_preds[0].cpu().squeeze().numpy()
plt.plot(t_tf, tf_plot, 'b--', label='Teacher Forcing (1-step ahead)', alpha=0.8)

# Autoregressive Prediction
plt.plot(t_ar, ar_preds, 'r:', label='Autoregressive (Free-running)', alpha=0.9, linewidth=2)

plt.axvline(x=SEQ_LEN-1, color='gray', linestyle=':', label='Prediction Start')
plt.title("Qualitative Results: Sequence Forecasting")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
save_path = "day107_gru_experiment.png"
plt.savefig(save_path, dpi=150)
print(f"\nPlot saved to {os.path.abspath(save_path)}")

# ==========================================
# GRU Gate Inspection (Sanity Check)
# ==========================================
print("\n--- Gate Activation Statistics (Validation Batch) ---")
model.eval()
with torch.no_grad():
    xb, _ = next(iter(val_loader))
    xb = xb.to(DEVICE)
    
    # Hook to capture gate values
    gate_vals = {}
    def hook_fn(name):
        def hook(module, input, output):
            gate_vals[name] = output.detach().cpu()
        return hook
    
    # PyTorch nn.GRU fuses gates, so we inspect the fused linear layer weights/biases 
    # or simply check hidden state saturation.
    # For a true gate inspection, a custom GRUCell is needed. 
    # Here we check hidden state distribution.
    _, hn = model(xb)
    h_final = hn.squeeze(0) # (Batch, Hidden)
    
    print(f"Hidden State Mean: {h_final.mean().item():.4f}, Std: {h_final.std().item():.4f}")
    print(f"Hidden State Min/Max: {h_final.min().item():.4f} / {h_final.max().item():.4f}")
    print(f"Saturation (>0.9 or <-0.9): {(h_final.abs() > 0.9).float().mean().item()*100:.2f}%")

print("\nExperiment Complete.")