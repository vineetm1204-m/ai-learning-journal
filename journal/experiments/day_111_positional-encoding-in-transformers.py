import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ============================================================
# Positional Encoding Implementations
# ============================================================

def sinusoidal_positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Original sinusoidal positional encoding from "Attention Is All You Need".
    PE(pos, 2i)   = sin(pos / base^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings (used in BERT, GPT, etc.)"""
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.embedding(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as a PyTorch module (no learnable params)"""
    def __init__(self, d_model: int, max_seq_len: int = 5000, base: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ============================================================
# Analysis & Visualization Functions
# ============================================================

def analyze_sinusoidal_properties(d_model: int = 512, seq_len: int = 100):
    """Analyze mathematical properties of sinusoidal encoding."""
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    
    print("=" * 60)
    print("SINUSOIDAL POSITIONAL ENCODING ANALYSIS")
    print("=" * 60)
    print(f"Sequence length: {seq_len}, d_model: {d_model}")
    print(f"Shape: {pe.shape}")
    print(f"Value range: [{pe.min():.4f}, {pe.max():.4f}]")
    print(f"Mean: {pe.mean():.6f}, Std: {pe.std():.6f}")
    
    # Check orthogonality between positions
    print("\n--- Position Similarity (cosine) ---")
    for delta in [1, 2, 4, 8, 16]:
        if delta < seq_len:
            sim = np.sum(pe[:-delta] * pe[delta:], axis=1) / (np.linalg.norm(pe[:-delta], axis=1) * np.linalg.norm(pe[delta:], axis=1))
            print(f"  Delta={delta:2d}: mean cos_sim={sim.mean():.4f}, std={sim.std():.4f}")
    
    # Check wavelength progression
    print("\n--- Wavelength Analysis (dimension pairs) ---")
    for i in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if i < d_model // 2:
            wavelength = 2 * np.pi * (10000 ** (2 * i / d_model))
            print(f"  Dim pair {i:3d}-{i+1:3d}: wavelength ≈ {wavelength:.1f}")
    
    return pe


def visualize_encodings(pe_sinusoidal: np.ndarray, pe_learned: Optional[np.ndarray] = None, 
                        seq_len: int = 50, d_model: int = 128):
    """Visualize positional encodings."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sinusoidal heatmap
    im1 = axes[0, 0].imshow(pe_sinusoidal[:seq_len, :d_model].T, aspect='auto', cmap='RdBu', 
                             vmin=-1, vmax=1)
    axes[0, 0].set_title('Sinusoidal Positional Encoding')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Sinusoidal: specific dimensions over positions
    dims_to_plot = [0, 1, 2, 3, 4, 5, 10, 20, 40, 80]
    dims_to_plot = [d for d in dims_to_plot if d < d_model]
    for d in dims_to_plot:
        axes[0, 1].plot(pe_sinusoidal[:seq_len, d], label=f'dim {d}', alpha=0.7)
    axes[0, 1].set_title('Sinusoidal: Dimension Values Across Positions')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Encoding Value')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learned encoding (if provided)
    if pe_learned is not None:
        im3 = axes[1, 0].imshow(pe_learned[:seq_len, :d_model].T, aspect='auto', cmap='RdBu',
                                 vmin=-1, vmax=1)
        axes[1, 0].set_title('Learned Positional Embedding (after init)')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Dimension')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 4. Comparison: position similarity matrices
        def compute_sim_matrix(pe):
            norm = np.linalg.norm(pe, axis=1, keepdims=True) + 1e-8
            return (pe / norm) @ (pe / norm).T
        
        sim_sin = compute_sim_matrix(pe_sinusoidal[:seq_len])
        sim_learned = compute_sim_matrix(pe_learned[:seq_len])
        
        im4a = axes[1, 1].imshow(sim_sin, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 1].set_title('Sinusoidal: Position Similarity Matrix')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Position')
        plt.colorbar(im4a, ax=axes[1, 1])
    else:
        # Just show similarity matrix for sinusoidal
        norm = np.linalg.norm(pe_sinusoidal[:seq_len], axis=1, keepdims=True) + 1e-8
        sim_sin = (pe_sinusoidal[:seq_len] / norm) @ (pe_sinusoidal[:seq_len] / norm).T
        im4 = axes[1, 0].imshow(sim_sin, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 0].set_title('Sinusoidal: Position Similarity Matrix')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Position')
        plt.colorbar(im4, ax=axes[1, 0])
        
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('positional_encoding_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to 'positional_encoding_analysis.png'")
    plt.close()


def demonstrate_relative_position_property(d_model: int = 512):
    """
    Demonstrate the key property: PE(pos + k) can be represented as 
    a linear function of PE(pos) for sinusoidal encodings.
    """
    print("\n" + "=" * 60)
    print("RELATIVE POSITION PROPERTY DEMONSTRATION")
    print("=" * 60)
    
    # For a fixed offset k, there exists a linear transformation T_k
    # such that PE(pos + k) = T_k @ PE(pos)
    # This is because sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
    
    k = 5  # offset
    seq_len = 20
    pe = sinusoidal_positional_encoding(seq_len + k, d_model)
    
    # Try to learn the linear transformation for each dimension pair
    # For each pair (2i, 2i+1), the transformation is a 2x2 rotation matrix
    print(f"Offset k = {k}")
    print("For each dimension pair (2i, 2i+1), the transformation is:")
    print("  [ cos(k * ω_i)  -sin(k * ω_i) ]")
    print("  [ sin(k * ω_i)   cos(k * ω_i) ]")
    print(f"  where ω_i = 1 / 10000^(2i/d_model)")
    
    # Verify for a few dimensions
    for i in [0, 1, 2, 10, 50, 100]:
        if 2*i + 1 < d_model:
            omega = 1 / (10000 ** (2 * i / d_model))
            cos_k = np.cos(k * omega)
            sin_k = np.sin(k * omega)
            
            # Check at a few positions
            errors = []
            for pos in range(5):
                if pos + k < seq_len:
                    x = pe[pos, 2*i:2*i+2]
                    x_next = pe[pos + k, 2*i:2*i+2]
                    T = np.array([[cos_k, -sin_k], [sin_k, cos_k]])
                    x_pred = T @ x
                    errors.append(np.linalg.norm(x_next - x_pred))
            print(f"  Dim pair {2*i:3d}-{2*i+1:3d}: ω={omega:.6f}, max error={max(errors):.2e}")


def toy_sequence_task():
    """
    Toy task: predict whether a sequence is in correct order or reversed.
    Shows that positional encoding is necessary for order awareness.
    """
    print("\n" + "=" * 60)
    print("TOY TASK: ORDER PREDICTION")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    vocab_size = 100
    d_model = 64
    seq_len = 10
    batch_size = 32
    n_epochs = 50
    
    # Generate data: sequences of random tokens, label = 1 if sorted, 0 if reversed
    def generate_batch(bsz):
        x = torch.randint(1, vocab_size, (bsz, seq_len))
        # Create sorted and reversed versions
        x_sorted, _ = torch.sort(x, dim=1)
        x_reversed = torch.flip(x_sorted, dims=[1])
        # Mix them
        labels = torch.randint(0, 2, (bsz,)).float()
        x_mixed = torch.where(labels.unsqueeze(1).bool(), x_sorted, x_reversed)
        return x_mixed, labels
    
    # Model without positional encoding
    class NoPEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
            self.classifier = nn.Linear(d_model, 1)
        
        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.attention(x, x, x)
            x = x.mean(dim=1)  # Global average pooling
            return self.classifier(x).squeeze(-1)
    
    # Model with sinusoidal PE
    class WithPEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pe = SinusoidalPositionalEncoding(d_model, max_seq_len=seq_len)
            self.attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
            self.classifier = nn.Linear(d_model, 1)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.pe(x)
            x, _ = self.attention(x, x, x)
            x = x.mean(dim=1)
            return self.classifier(x).squeeze(-1)
    
    # Model with learned PE
    class LearnedPEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pe = LearnedPositionalEncoding(seq_len, d_model)
            self.attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
            self.classifier = nn.Linear(d_model, 1)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.pe(x)
            x, _ = self.attention(x, x, x)
            x = x.mean(dim=1)
            return self.classifier(x).squeeze(-1)
    
    models = {
        'No PE': NoPEModel(),
        'Sinusoidal PE': WithPEModel(),
        'Learned PE': LearnedPEModel(),
    }
    
    results = {}
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        losses = []
        accuracies = []
        
        for epoch in range(n_epochs):
            model.train()
            x_batch, y_batch = generate_batch(batch_size)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                x_test, y_test = generate_batch(200)
                logits_test = model(x_test)
                preds = (torch.sigmoid(logits_test) > 0.5).float()
                acc = (preds == y_test).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(acc)
        
        results[name] = {'losses': losses, 'accuracies': accuracies}
        print(f"{name:20s}: Final Acc = {accuracies[-1]:.4f}, Final Loss = {losses[-1]:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, res in results.items():
        axes[0].plot(res['losses'], label=name, alpha=0.8)
        axes[1].plot(res['accuracies'], label=name, alpha=0.8)
    
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('toy_task_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved toy task results to 'toy_task_results.png'")
    plt.close()
    
    return results


def attention_pattern_analysis():
    """Analyze how positional encoding affects attention patterns."""
    print("\n" + "=" * 60)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 60)
    
    d_model = 64
    seq_len = 8
    n_heads = 4
    
    # Create a simple sequence
    x = torch.randn(1, seq_len, d_model)
    
    # Add positional encodings
    pe_sin = SinusoidalPositionalEncoding(d_model, max_seq_len=seq_len)
    pe_learned = LearnedPositionalEncoding(seq_len, d_model)
    
    x_sin = pe_sin(x.clone())
    x_learned = pe_learned(x.clone())
    
    # Compute attention scores (simplified: Q=K=x, no projection)
    def compute_attention(x):
        # x: [1, seq_len, d_model]
        scores = torch.matmul(x, x.transpose(-2, -1)) / np.sqrt(d_model)
        attn = F.softmax(scores, dim=-1)
        return attn.squeeze(0).detach().numpy()
    
    attn_no_pe = compute_attention(x)
    attn_sin = compute_attention(x_sin)
    attn_learned = compute_attention(x_learned)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, attn, title in zip(axes, [attn_no_pe, attn_sin, attn_learned], 
                                ['No PE', 'Sinusoidal PE', 'Learned PE']):
        im = ax.imshow(attn, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    print("Saved attention patterns to 'attention_patterns.png'")
    plt.close()
    
    # Analyze diagonal dominance (self-attention)
    for name, attn in [('No PE', attn_no_pe), ('Sinusoidal', attn_sin), ('Learned', attn_learned)]:
        diag = np.diag(attn).mean()
        off_diag = (attn.sum() - np.diag(attn).sum()) / (seq_len * (seq_len - 1))
        print(f"  {name:12s}: Diagonal mean={diag:.4f}, Off-diagonal mean={off_diag:.4f}")


def extrapolation_test():
    """Test how well sinusoidal vs learned PE extrapolate to longer sequences."""
    print("\n" + "=" * 60)
    print("EXTRAPOLATION TEST (Train on len=64, Test on len=128)")
    print("=" * 60)
    
    torch.manual_seed(123)
    
    d_model = 128
    train_len = 64
    test_len = 128
    vocab_size = 1000
    
    # Task: predict next token in a simple pattern (e.g., arithmetic progression)
    def generate_arithmetic_batch(bsz, seq_len):
        # Start with random number, add random step
        starts = torch.randint(0, 100, (bsz, 1))
        steps = torch.randint(1, 10, (bsz, 1))
        positions = torch.arange(seq_len).float().unsqueeze(0)
        sequences = starts + steps * positions
        # Quantize to vocab
        sequences = (sequences * 10).long() % vocab_size
        # Target is next token
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        return inputs, targets
    
    class TransformerModel(nn.Module):
        def __init__(self, pe_type='sinusoidal'):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            if pe_type == 'sinusoidal':
                self.pe = SinusoidalPositionalEncoding(d_model, max_seq_len=test_len)
            elif pe_type == 'learned':
                self.pe = LearnedPositionalEncoding(test_len, d_model)
            else:
                self.pe = None
            
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=256, 
                                                        batch_first=True, dropout=0.1)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            if self.pe is not None:
                x = self.pe(x)
            x = self.transformer(x)
            return self.output(x)
    
    for pe_type in ['none', 'sinusoidal', 'learned']:
        model = TransformerModel(pe_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        model.train()
        for epoch in range(30):
            x_batch, y_batch = generate_arithmetic_batch(32, train_len)
            logits = model(x_batch)
            loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Test on training length
        model.eval()
        with torch.no_grad():
            x_test, y_test = generate_arithmetic_batch(100, train_len)
            logits = model(x_test)
            preds = logits.argmax(dim=-1)
            train_acc = (preds == y_test).float().mean().item()
            
            # Test on longer sequence (extrapolation)
            x_test_long, y_test_long = generate_arithmetic_batch(100, test_len)
            logits_long = model(x_test_long)
            preds_long = logits_long.argmax(dim=-1)
            test_acc = (preds_long == y_test_long).float().mean().item()
        
        print(f"  {pe_type:12s}: Train len acc={train_acc:.4f}, Extrapolation acc={test_acc:.4f}")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 111: POSITIONAL ENCODING IN TRANSFORMERS")
    print("=" * 60)
    
    # 1. Analyze sinusoidal encoding properties
    pe_sin = analyze_sinusoidal_properties(d_model=512, seq_len=100)
    
    # 2. Demonstrate relative position property
    demonstrate_relative_position_property(d_model=512)
    
    # 3. Create learned embedding for comparison
    learned_pe_module = LearnedPositionalEncoding(100, 128)
    with torch.no_grad():
        pos = torch.arange(100).unsqueeze(0)
        pe_learned = learned_pe_module.embedding(pos).squeeze(0).numpy()
    
    # 4. Visualize
    visualize_encodings(pe_sin, pe_learned, seq_len=50, d_model=128)
    
    # 5. Toy task showing necessity of PE
    toy_sequence_task()
    
    # 6. Attention pattern analysis
    attention_pattern_analysis()
    
    # 7. Extrapolation test
    extrapolation_test()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("Generated files:")
    print("  - positional_encoding_analysis.png")
    print("  - toy_task_results.png")
    print("  - attention_patterns.png")