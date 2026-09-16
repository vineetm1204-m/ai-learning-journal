import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
    def forward(self, x):
        return self.pe[:x.size(1), :].unsqueeze(0)

def visualize_encodings(pe_sin, pe_learned, max_len, d_model):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(pe_sin.numpy().T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title('Sinusoidal PE (dim x pos)')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')
    plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
    
    axes[0, 1].imshow(pe_learned.squeeze(0).detach().numpy().T, aspect='auto', cmap='RdBu')
    axes[0, 1].set_title('Learned PE (dim x pos)')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Dimension')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    
    for i in [0, 1, 2, 4, 8, 16]:
        if i < d_model:
            axes[1, 0].plot(range(max_len), pe_sin[:, i].numpy(), label=f'dim {i}')
    axes[1, 0].set_title('Sinusoidal PE: Dimensions vs Position')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for i in [0, 1, 2, 4, 8, 16]:
        if i < d_model:
            axes[1, 1].plot(range(max_len), pe_learned.squeeze(0)[:, i].detach().numpy(), label=f'dim {i}')
    axes[1, 1].set_title('Learned PE: Dimensions vs Position')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png', dpi=150)
    plt.close()

def test_relative_position_property(pe, max_len, d_model):
    print("\n=== Relative Position Property Test ===")
    for offset in [1, 2, 4, 8]:
        dots = []
        for pos in range(max_len - offset):
            dot = torch.dot(pe[pos], pe[pos + offset])
            dots.append(dot.item())
        avg_dot = np.mean(dots)
        std_dot = np.std(dots)
        print(f"Offset {offset:2d}: mean dot={avg_dot:.4f}, std={std_dot:.4f}")

def test_interpolation(pe_sin, max_len, d_model):
    print("\n=== Interpolation Test (Extrapolation) ===")
    new_len = max_len * 2
    pe_extended = sinusoidal_positional_encoding(new_len, d_model)
    overlap = pe_extended[:max_len]
    diff = torch.mean((pe_sin - overlap) ** 2).item()
    print(f"MSE between original and extended (first {max_len} positions): {diff:.6f}")
    print(f"Extrapolation works: {diff < 1e-10}")

def simple_task_comparison():
    print("\n=== Simple Sequence Task: Position Classification ===")
    max_len, d_model = 32, 64
    pe_sin = sinusoidal_positional_encoding(max_len, d_model)
    pe_learned = LearnedPositionalEncoding(max_len, d_model)
    
    class PositionClassifier(nn.Module):
        def __init__(self, pe_type):
            super().__init__()
            self.pe_type = pe_type
            if pe_type == 'sinusoidal':
                self.register_buffer('pe', pe_sin)
            else:
                self.pe_module = pe_learned
            self.classifier = nn.Linear(d_model, max_len)
        
        def forward(self, x):
            if self.pe_type == 'sinusoidal':
                pe = self.pe[:x.size(1)]
            else:
                pe = self.pe_module(x)
            return self.classifier(x + pe)
    
    x = torch.randn(100, max_len, d_model)
    y = torch.arange(max_len).repeat(100, 1)
    
    for pe_type in ['sinusoidal', 'learned']:
        model = PositionClassifier(pe_type)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(50):
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, max_len), y.view(-1))
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
            acc = (logits.argmax(-1) == y).float().mean().item()
        print(f"{pe_type:12s}: Final Accuracy = {acc:.4f}")

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    max_len, d_model = 100, 512
    
    print("=" * 60)
    print("DAY 70: Positional Encoding in Transformers")
    print("=" * 60)
    
    pe_sin = sinusoidal_positional_encoding(max_len, d_model)
    pe_learned = LearnedPositionalEncoding(max_len, d_model)
    
    print(f"\nSinusoidal PE shape: {pe_sin.shape}")
    print(f"Learned PE shape: {pe_learned.pe.shape}")
    print(f"Sinusoidal PE range: [{pe_sin.min():.3f}, {pe_sin.max():.3f}]")
    print(f"Learned PE range: [{pe_learned.pe.min():.3f}, {pe_learned.pe.max():.3f}]")
    
    test_relative_position_property(pe_sin, max_len, d_model)
    test_relative_position_property(pe_learned.pe.squeeze(0), max_len, d_model)
    
    test_interpolation(pe_sin, max_len, d_model)
    
    visualize_encodings(pe_sin, pe_learned, max_len, min(d_model, 64))
    print("\nVisualization saved to 'positional_encoding_visualization.png'")
    
    simple_task_comparison()
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("1. Sinusoidal PE has deterministic relative position structure")
    print("2. Learned PE adapts but requires training data")
    print("3. Sinusoidal PE extrapolates naturally to longer sequences")
    print("4. Both enable position awareness in permutation-invariant attention")
    print("=" * 60)

if __name__ == "__main__":
    main()