import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Day 25: GRU Mini-Experiment
# Task: Sequence Classification (Parity Check)
# Input:  Binary sequences of variable length.
# Target: 1 if number of 1s is odd, 0 if even.
# Goal:   Demonstrate GRU gating mechanics (Update/Reset gates)
#         and compare vs Vanilla RNN.
# ============================================================

# --- Config ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
SEQ_LEN = 20
HIDDEN_SIZE = 32
EPOCHS = 15
LR = 0.01

# --- 1. Synthetic Data Generator ---
def generate_parity_data(n_samples, seq_len):
    """Generates binary sequences and parity labels."""
    X = torch.randint(0, 2, (n_samples, seq_len, 1), dtype=torch.float32)
    # Parity: sum mod 2
    y = (X.sum(dim=1) % 2).long().squeeze()
    return X, y

train_X, train_y = generate_parity_data(5000, SEQ_LEN)
test_X, test_y = generate_parity_data(1000, SEQ_LEN)

train_ds = torch.utils.data.TensorDataset(train_X, train_y)
test_ds = torch.utils.data.TensorDataset(test_X, test_y)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- 2. Custom GRU Cell (Educational: Exposes Gates) ---
class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Update gate: z = sigmoid(W_z [h, x])
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # Reset gate: r = sigmoid(W_r [h, x])
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate hidden: h_tilde = tanh(W_h [r*h, x])
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h, return_gates=False):
        # x: (batch, input_size), h: (batch, hidden_size)
        combined = torch.cat([h, x], dim=1)
        
        z = torch.sigmoid(self.W_z(combined))      # Update gate
        r = torch.sigmoid(self.W_r(combined))      # Reset gate
        
        combined_reset = torch.cat([r * h, x], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset)) # Candidate
        
        h_next = (1 - z) * h + z * h_tilde         # Final state
        
        if return_gates:
            return h_next, {'z': z, 'r': r, 'h_tilde': h_tilde}
        return h_next

# --- 3. Models ---
class GRUClassifier(nn.Module):
    def __init__(self, cell_type='gru'):
        super().__init__()
        self.cell_type = cell_type
        if cell_type == 'gru':
            self.rnn = nn.GRU(1, HIDDEN_SIZE, batch_first=True)
        elif cell_type == 'custom_gru':
            self.cell = CustomGRUCell(1, HIDDEN_SIZE)
        elif cell_type == 'vanilla':
            self.rnn = nn.RNN(1, HIDDEN_SIZE, batch_first=True, nonlinearity='tanh')
        else:
            raise ValueError("Unknown type")
        self.fc = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, x, analyze_gates=False):
        if self.cell_type == 'custom_gru':
            h = torch.zeros(x.size(0), HIDDEN_SIZE, device=x.device)
            gate_history = []
            for t in range(x.size(1)):
                h, gates = self.cell(x[:, t, :], h, return_gates=True)
                if analyze_gates: gate_history.append(gates)
            out = self.fc(h)
            return out, gate_history
        else:
            _, h_n = self.rnn(x)
            out = self.fc(h_n.squeeze(0))
            return out, None

# --- 4. Training Loop ---
def train_model(model, loader, criterion, optimizer, device, name):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total

# --- 5. Gate Analysis Utility ---
def analyze_gate_saturation(gate_history):
    """Prints mean activation stats for gates over the sequence."""
    if not gate_history: return
    z_vals = torch.cat([g['z'] for g in gate_history], dim=0)
    r_vals = torch.cat([g['r'] for g in gate_history], dim=0)
    print(f"  Gate Stats -> Update(z) mean: {z_vals.mean():.3f} (saturation: {(z_vals>0.9).float().mean():.2%}) | "
          f"Reset(r) mean: {r_vals.mean():.3f} (saturation: {(r_vals>0.9).float().mean():.2%})")

# --- 6. Main Experiment Runner ---
def run_experiment():
    print(f"Device: {DEVICE} | SeqLen: {SEQ_LEN} | Hidden: {HIDDEN_SIZE}")
    criterion = nn.CrossEntropyLoss()
    results = {}

    for model_name in ['vanilla', 'gru', 'custom_gru']:
        print(f"\n--- Training {model_name.upper()} ---")
        model = GRUClassifier(model_name).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        history = {'train_acc': [], 'test_acc': []}
        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_model(model, train_loader, criterion, optimizer, DEVICE, model_name)
            te_loss, te_acc = eval_model(model, test_loader, criterion, DEVICE)
            history['train_acc'].append(tr_acc)
            history['test_acc'].append(te_acc)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Ep {epoch:2d} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")
        
        results[model_name] = history

        # Gate Analysis for Custom GRU on a test batch
        if model_name == 'custom_gru':
            print("\n--- Gate Dynamics Analysis (Test Batch) ---")
            model.eval()
            with torch.no_grad():
                X_batch, _ = next(iter(test_loader))
                X_batch = X_batch.to(DEVICE)
                _, gates = model(X_batch, analyze_gates=True)
                analyze_gate_saturation(gates)

    # --- 7. Visualization (Text-based Plot) ---
    print("\n\n=== FINAL COMPARISON (Test Accuracy) ===")
    epochs_range = range(1, EPOCHS + 1)
    # Simple ASCII plot
    width = 50
    for name in ['vanilla', 'gru', 'custom_gru']:
        accs = results[name]['test_acc']
        print(f"\n{name.upper():<12}: ", end="")
        for i, acc in enumerate(accs):
            bar_len = int(acc * width)
            if i == EPOCHS - 1:
                print(f"[{'#'*bar_len}{'.'*(width-bar_len)}] {acc:.4f}")
            elif i % 3 == 0:
                print(f"[{'#'*bar_len}{'.'*(width-bar_len)}]", end=" ")

    print("\n\nExperiment Complete.")

if __name__ == "__main__":
    run_experiment()