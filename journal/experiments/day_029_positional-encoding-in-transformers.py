import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Positional Encoding Implementations
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard 'Attention Is All You Need' sinusoidal encoding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    """Learnable absolute positional embeddings (BERT/GPT style)."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class RoPEPositionalEncoding(nn.Module):
    """Rotary Positional Embedding (RoPE) - applies rotation to query/key vectors.
    This is a simplified application logic for visualization.
    """
    def __init__(self, d_model: int, max_len: int = 2048, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        # Precompute frequencies (inverse frequencies)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def _get_cos_sin(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Shape: (seq_len, d_model/2) -> cat for full d_model
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RoPE to input tensor x (batch, seq_len, d_model)."""
        batch, seq_len, d_model = x.shape
        cos, sin = self._get_cos_sin(seq_len, x.device)
        
        # Reshape for broadcasting: (1, seq_len, d_model)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        
        # Split last dim into pairs (x1, x2)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Rotate: [x1, x2] * [cos, -sin; sin, cos]
        # Result: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
        rotated_x1 = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        rotated_x2 = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        
        # Interleave back
        out = torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)
        return out


# ============================================================
# 2. Analysis & Visualization Utilities
# ============================================================

def visualize_heatmap(pe_matrix: np.ndarray, title: str, ax=None):
    """Plots the positional encoding matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pe_matrix.T, aspect='auto', cmap='RdBu', 
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Position (Sequence Length)")
    ax.set_ylabel("Dimension (d_model)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def calculate_positional_similarity(pe_matrix: np.ndarray):
    """Calculates cosine similarity between positional vectors."""
    # Normalize rows
    norms = np.linalg.norm(pe_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = pe_matrix / norms
    sim_matrix = normalized @ normalized.T
    return sim_matrix

def plot_similarity(sim_matrix: np.ndarray, title: str, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def test_rope_relative_property(rope: RoPEPositionalEncoding, d_model=64):
    """Verifies RoPE property: Attention(q_m, k_n) depends on (m-n)."""
    print("\n--- RoPE Relative Position Property Test ---")
    # Create dummy queries/keys at different positions
    # We simulate the dot product <R_m q, R_n k> = q^T R_{m-n} k
    # For simplicity, use random vectors q, k and check if dot product 
    # matches for pairs with same relative distance.
    
    q = torch.randn(1, 1, d_model)
    k = torch.randn(1, 1, d_model)
    
    # Apply RoPE at pos 10 and 15 (dist 5)
    rope_q_10 = rope.apply_rope(q.repeat(1, 11, 1))[:, 10:11, :]
    rope_k_15 = rope.apply_rope(k.repeat(1, 16, 1))[:, 15:16, :]
    dot_10_15 = (rope_q_10 @ rope_k_15.transpose(-2, -1)).item()
    
    # Apply RoPE at pos 20 and 25 (dist 5)
    rope_q_20 = rope.apply_rope(q.repeat(1, 21, 1))[:, 20:21, :]
    rope_k_25 = rope.apply_rope(k.repeat(1, 26, 1))[:, 25:26, :]
    dot_20_25 = (rope_q_20 @ rope_k_25.transpose(-2, -1)).item()
    
    # Apply RoPE at pos 10 and 20 (dist 10)
    rope_k_20 = rope.apply_rope(k.repeat(1, 21, 1))[:, 20:21, :]
    dot_10_20 = (rope_q_10 @ rope_k_20.transpose(-2, -1)).item()
    
    print(f"Dot Product (Pos 10, 15) dist=5: {dot_10_15:.6f}")
    print(f"Dot Product (Pos 20, 25) dist=5: {dot_20_25:.6f}")
    print(f"Dot Product (Pos 10, 20) dist=10: {dot_10_20:.6f}")
    print(f"Diff (dist 5 pairs): {abs(dot_10_15 - dot_20_25):.6f} (Should be near 0)")
    assert abs(dot_10_15 - dot_20_25) < 1e-4, "RoPE Relative Property Failed!"
    print("✅ RoPE Relative Position Invariance Verified.")


# ============================================================
# 3. Main Experiment Runner
# ============================================================

def run_experiment():
    print("=" * 60)
    print("Day 29: Positional Encoding in Transformers - Mini Experiment")
    print("=" * 60)
    
    CONFIG = {
        "d_model": 128,
        "seq_len": 100,
        "max_len": 200
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # --- 1. Sinusoidal (Fixed) ---
    print("\n[1/4] Analyzing Sinusoidal Positional Encoding...")
    sin_pe = SinusoidalPositionalEncoding(CONFIG["d_model"], CONFIG["max_len"]).to(device)
    dummy_input = torch.zeros(1, CONFIG["seq_len"], CONFIG["d_model"]).to(device)
    sin_out = sin_pe(dummy_input).squeeze(0).cpu().numpy() # (seq_len, d_model)
    
    # --- 2. Learnable ---
    print("[2/4] Analyzing Learnable Positional Encoding...")
    learn_pe = LearnablePositionalEncoding(CONFIG["d_model"], CONFIG["max_len"]).to(device)
    learn_out = learn_pe(dummy_input).squeeze(0).detach().cpu().numpy()
    
    # --- 3. RoPE ---
    print("[3/4] Analyzing Rotary Positional Encoding (RoPE)...")
    rope = RoPEPositionalEncoding(CONFIG["d_model"], CONFIG["max_len"]).to(device)
    # RoPE modifies vectors, so we visualize the rotation effect on a basis vector
    basis = torch.eye(CONFIG["d_model"]).unsqueeze(0).repeat(CONFIG["seq_len"], 1, 1).to(device) # (seq, d, d)
    # Apply rope to each position's identity basis to see transformation
    # Actually, simpler: apply to a constant vector across positions
    const_vec = torch.ones(1, CONFIG["seq_len"], CONFIG["d_model"]).to(device)
    rope_out = rope.apply_rope(const_vec).squeeze(0).cpu().numpy()
    
    # --- 4. Visualization ---
    print("[4/4] Generating Visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    
    # Row 0: Sinusoidal
    visualize_heatmap(sin_out, "Sinusoidal PE (Fixed)", axes[0, 0])
    sin_sim = calculate_positional_similarity(sin_out)
    plot_similarity(sin_sim, "Sinusoidal: Cosine Similarity (Position vs Position)", axes[0, 1])
    
    # Row 1: Learnable
    visualize_heatmap(learn_out, "Learnable PE (Random Init)", axes[1, 0])
    learn_sim = calculate_positional_similarity(learn_out)
    plot_similarity(learn_sim, "Learnable: Cosine Similarity", axes[1, 1])
    
    # Row 2: RoPE
    visualize_heatmap(rope_out, "RoPE: Rotated Constant Vector", axes[2, 0])
    rope_sim = calculate_positional_similarity(rope_out)
    plot_similarity(rope_sim, "RoPE: Cosine Similarity", axes[2, 1])
    
    plt.tight_layout()
    plt.savefig("day29_positional_encoding_analysis.png", dpi=150)
    print("📸 Saved visualization to 'day29_positional_encoding_analysis.png'")
    
    # --- 5. Quantitative Comparison ---
    print("\n--- Quantitative Comparison ---")
    def stats(name, mat):
        sim = calculate_positional_similarity(mat)
        # Avg similarity of adjacent positions vs distant
        adj_sim = np.mean([sim[i, i+1] for i in range(CONFIG["seq_len"]-1)])
        dist_sim = np.mean([sim[i, i+20] for i in range(CONFIG["seq_len"]-20)])
        print(f"{name:20s} | Adjacent Sim: {adj_sim:.4f} | Distant (20) Sim: {dist_sim:.4f} | Diag Mean: {np.trace(sim)/CONFIG['seq_len']:.4f}")
    
    stats("Sinusoidal", sin_out)
    stats("Learnable", learn_out)
    stats("RoPE", rope_out)
    
    # --- 6. RoPE Property Verification ---
    test_rope_relative_property(rope, CONFIG["d_model"])
    
    print("\n" + "=" * 60)
    print("Experiment Complete.")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()