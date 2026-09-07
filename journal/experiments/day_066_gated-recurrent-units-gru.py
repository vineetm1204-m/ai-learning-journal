import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SEQ_LENGTH = 20
INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 1. Synthetic Data Generation: Noisy Sine Wave ---
def generate_data(num_samples=1000):
    t = np.linspace(0, 4 * np.pi, num_samples + SEQ_LENGTH)
    # Clean signal
    signal = np.sin(t)
    # Add noise
    noise = 0.1 * np.random.randn(len(t))
    noisy_signal = signal + noise
    
    X, y = [], []
    for i in range(num_samples):
        X.append(noisy_signal[i : i + SEQ_LENGTH])
        y.append(signal[i + SEQ_LENGTH]) # Predict next step of CLEAN signal
        
    X = np.array(X).reshape(-1, SEQ_LENGTH, INPUT_SIZE).astype(np.float32)
    y = np.array(y).reshape(-1, OUTPUT_SIZE).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

X, y = generate_data(2000)
split = int(0.8 * len(X))
train_data = TensorDataset(X[:split], y[:split])
val_data = TensorDataset(X[split:], y[split:])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. GRU Model Definition ---
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True expects (batch, seq, feature)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0 shape: (num_layers, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        # out shape: (batch, seq_length, hidden_size)
        out, _ = self.gru(x, h0)
        # Take only the last time step output
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = GRURegressor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- 3. Training Loop ---
print(f"Training on {DEVICE}...")
history = {'train_loss': [], 'val_loss': []}

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        train_losses.append(loss.item())
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())
            
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    history['train_loss'].append(avg_train)
    history['val_loss'].append(avg_val)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

# --- 4. Qualitative Evaluation: Autoregressive Prediction ---
model.eval()
# Seed with a sequence from validation set
seed_idx = 0
input_seq = X[split + seed_idx].unsqueeze(0).to(DEVICE) # (1, SEQ, 1)
true_future = signal = np.sin(np.linspace(0, 4*np.pi, 2000 + SEQ_LENGTH))[split + seed_idx + SEQ_LENGTH : split + seed_idx + SEQ_LENGTH + 50]

predictions = []
current_seq = input_seq.clone()

with torch.no_grad():
    for _ in range(50):
        pred = model(current_seq) # (1, 1)
        predictions.append(pred.item())
        # Roll sequence: drop first, append prediction
        current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(1).unsqueeze(2)], dim=1)

# --- 5. Plotting ---
plt.figure(figsize=(12, 5))

# Loss Curves
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('GRU Training Dynamics (Day 66)')
plt.legend()
plt.grid(True, alpha=0.3)

# Prediction vs Ground Truth
plt.subplot(1, 2, 2)
plt.plot(range(SEQ_LENGTH), input_seq[0].cpu().numpy().flatten(), 'b-', label='Input Sequence (Noisy)', alpha=0.6)
plt.plot(range(SEQ_LENGTH, SEQ_LENGTH + 50), true_future, 'g-', label='Ground Truth (Clean)', linewidth=2)
plt.plot(range(SEQ_LENGTH, SEQ_LENGTH + 50), predictions, 'r--', label='GRU Autoregressive Pred', linewidth=2)
plt.axvline(x=SEQ_LENGTH-1, color='k', linestyle=':', label='Prediction Start')
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.title('Denoising & Forecasting Demo')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
save_path = "day66_gru_experiment.png"
plt.savefig(save_path, dpi=150)
print(f"\nPlot saved to {os.path.abspath(save_path)}")
print("Experiment Complete.")