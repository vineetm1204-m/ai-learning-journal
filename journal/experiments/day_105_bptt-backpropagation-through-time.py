import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

torch.manual_seed(42)
np.random.seed(42)

class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((x, hidden), dim=1)
        hidden = self.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size)

def generate_sequence_data(seq_len: int, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0, 4 * np.pi, seq_len)
    data = torch.sin(t).unsqueeze(1).repeat(1, n_samples).t()
    targets = torch.roll(data, shifts=-1, dims=1)
    targets[:, -1] = torch.sin(t[0] + (t[1] - t[0]))
    return data.unsqueeze(-1), targets.unsqueeze(-1)

def train_bptt(model: SimpleRNN, inputs: torch.Tensor, targets: torch.Tensor, 
               seq_len: int, epochs: int, lr: float, clip_grad: float = None,
               truncate: int = None) -> List[float]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        hidden = model.init_hidden(inputs.size(0))
        
        for t in range(seq_len):
            x = inputs[:, t:t+1, :]
            y = targets[:, t:t+1, :]
            
            output, hidden = model(x.squeeze(1), hidden)
            loss = criterion(output.unsqueeze(1), y)
            total_loss += loss.item()
            
            if truncate and (t + 1) % truncate == 0:
                loss.backward(retain_graph=True)
                if clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                hidden = hidden.detach()
            elif t == seq_len - 1:
                loss.backward()
                if clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()
        
        losses.append(total_loss / seq_len)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]:.6f}")
    
    return losses

def analyze_gradient_flow(model: SimpleRNN, inputs: torch.Tensor, targets: torch.Tensor, 
                          seq_len: int) -> List[float]:
    model.eval()
    criterion = nn.MSELoss()
    hidden = model.init_hidden(inputs.size(0))
    grad_norms = []
    
    for t in range(seq_len):
        x = inputs[:, t:t+1, :]
        y = targets[:, t:t+1, :]
        output, hidden = model(x.squeeze(1), hidden)
        loss = criterion(output.unsqueeze(1), y)
        
        model.zero_grad()
        loss.backward(retain_graph=True)
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(np.sqrt(total_norm))
    
    return grad_norms

def visualize_bptt():
    seq_len = 50
    n_samples = 32
    hidden_size = 64
    epochs = 200
    lr = 0.01
    
    inputs, targets = generate_sequence_data(seq_len, n_samples)
    
    print("=" * 60)
    print("BPTT Experiment: Full vs Truncated vs Gradient Clipping")
    print("=" * 60)
    
    print("\n1. Full BPTT (no truncation)")
    model_full = SimpleRNN(1, hidden_size, 1)
    losses_full = train_bptt(model_full, inputs, targets, seq_len, epochs, lr)
    
    print("\n2. Truncated BPTT (truncate=10)")
    model_trunc = SimpleRNN(1, hidden_size, 1)
    losses_trunc = train_bptt(model_trunc, inputs, targets, seq_len, epochs, lr, truncate=10)
    
    print("\n3. Full BPTT with Gradient Clipping (max_norm=1.0)")
    model_clip = SimpleRNN(1, hidden_size, 1)
    losses_clip = train_bptt(model_clip, inputs, targets, seq_len, epochs, lr, clip_grad=1.0)
    
    print("\n4. Gradient Flow Analysis (Full BPTT)")
    model_grad = SimpleRNN(1, hidden_size, 1)
    _ = train_bptt(model_grad, inputs, targets, seq_len, 50, lr)
    grad_norms = analyze_gradient_flow(model_grad, inputs, targets, seq_len)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(losses_full, label='Full BPTT', alpha=0.8)
    axes[0, 0].plot(losses_trunc, label='Truncated BPTT (k=10)', alpha=0.8)
    axes[0, 0].plot(losses_clip, label='Full BPTT + Clipping', alpha=0.8)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(grad_norms, 'o-', color='red', alpha=0.7)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Gradient Norm (log scale)')
    axes[0, 1].set_title('Gradient Norm Flow Through Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    model_full.eval()
    with torch.no_grad():
        hidden = model_full.init_hidden(1)
        test_input = inputs[0:1]
        preds = []
        for t in range(seq_len):
            out, hidden = model_full(test_input[:, t:t+1, :].squeeze(1), hidden)
            preds.append(out.item())
    
    axes[1, 0].plot(targets[0, :, 0].numpy(), label='Target (sin)', alpha=0.7)
    axes[1, 0].plot(preds, label='Predicted', alpha=0.7, linestyle='--')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Sequence Prediction (Full BPTT)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(losses_full[:50], label='Full BPTT', alpha=0.8)
    axes[1, 1].plot(losses_trunc[:50], label='Truncated BPTT', alpha=0.8)
    axes[1, 1].plot(losses_clip[:50], label='Full BPTT + Clipping', alpha=0.8)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('Early Training Dynamics (First 50 Epochs)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bptt_experiment.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'bptt_experiment.png'")
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("=" * 60)
    print(f"Final Loss - Full BPTT: {losses_full[-1]:.6f}")
    print(f"Final Loss - Truncated BPTT: {losses_trunc[-1]:.6f}")
    print(f"Final Loss - Clipped BPTT: {losses_clip[-1]:.6f}")
    print(f"\nGradient norm range: [{min(grad_norms):.2e}, {max(grad_norms):.2e}]")
    print(f"Gradient norm ratio (max/min): {max(grad_norms)/min(grad_norms):.2f}")
    print("\nBPTT unrolls the computational graph through time.")
    print("Truncated BPTT limits memory but may lose long-term dependencies.")
    print("Gradient clipping prevents exploding gradients in deep time.")

if __name__ == "__main__":
    visualize_bptt()