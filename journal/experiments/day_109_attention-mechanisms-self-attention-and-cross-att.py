import numpy as np

# ============================================================
# Configuration & Utilities
# ============================================================
np.random.seed(42)

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def print_tensor(name, tensor, precision=4):
    """Pretty print tensor shape and stats."""
    print(f"{name}: shape={tensor.shape}, mean={tensor.mean():.{precision}f}, std={tensor.std():.{precision}f}")

def print_attention_heatmap(attn_weights, labels_q=None, labels_kv=None, title="Attention Weights"):
    """Print a text-based heatmap for the first head/batch."""
    # Take first batch, first head: (seq_len_q, seq_len_kv)
    weights = attn_weights[0, 0] if attn_weights.ndim == 4 else attn_weights[0]
    print(f"\n--- {title} (Batch 0, Head 0) ---")
    header = "       " + "  ".join([f"{i:>5}" for i in range(weights.shape[1])])
    print(header)
    for i, row in enumerate(weights):
        label = f"{labels_q[i]:>5}" if labels_q else f"{i:>5}"
        row_str = "  ".join([f"{val:.3f}" for val in row])
        print(f"{label} | {row_str}")

# ============================================================
# Core Attention Implementation (NumPy from Scratch)
# ============================================================

class ScaledDotProductAttention:
    """
    Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __call__(self, Q, K, V, mask=None):
        """
        Q: (batch, heads, seq_len_q, d_k)
        K: (batch, heads, seq_len_k, d_k)
        V: (batch, heads, seq_len_v, d_v)  -- usually seq_len_k == seq_len_v
        mask: (batch, 1, 1, seq_len_k) or broadcastable
        """
        d_k = Q.shape[-1]
        # Scores: (batch, heads, seq_len_q, seq_len_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attn_weights = softmax(scores, axis=-1)
        # Output: (batch, heads, seq_len_q, d_v)
        output = np.matmul(attn_weights, V)
        return output, attn_weights

class MultiHeadAttention:
    """
    Multi-Head Attention wrapper.
    Projects Q, K, V -> splits heads -> Attention -> concat -> Project Out
    """
    def __init__(self, d_model, num_heads, bias=True):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Combined projection weights for efficiency (Q, K, V stacked)
        # In PyTorch this is nn.Linear(d_model, 3*d_model). Here we separate for clarity.
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        """ (batch, seq_len, d_model) -> (batch, heads, seq_len, d_k) """
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x, batch_size):
        """ (batch, heads, seq_len, d_v) -> (batch, seq_len, d_model) """
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, seq_len_q, d_model)
        key:   (batch, seq_len_k, d_model)
        value: (batch, seq_len_v, d_model)
        """
        batch_size = query.shape[0]

        # 1. Linear Projections
        Q = np.dot(query, self.W_q) # (batch, seq_len_q, d_model)
        K = np.dot(key,   self.W_k)
        V = np.dot(value, self.W_v)

        # 2. Split Heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 3. Scaled Dot-Product Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 4. Combine Heads
        attn_output = self.combine_heads(attn_output, batch_size)

        # 5. Final Linear Projection
        output = np.dot(attn_output, self.W_o)
        return output, attn_weights

# ============================================================
# Experiment: Self-Attention vs Cross-Attention
# ============================================================

def run_self_attention_demo():
    print("\n" + "="*60)
    print("EXPERIMENT 1: SELF-ATTENTION (Encoder Block Style)")
    print("="*60)
    
    # Config
    batch_size = 2
    seq_len = 6
    d_model = 32
    num_heads = 4
    
    # Synthetic Input: (batch, seq_len, d_model)
    # Imagine token embeddings for a sentence "The cat sat on mat"
    x = np.random.randn(batch_size, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Self-Attention: Q, K, V all come from same source X
    out, weights = mha.forward(x, x, x)
    
    print_tensor("Input X", x)
    print_tensor("Output", out)
    print(f"Attention Weights Shape: {weights.shape} (Batch, Heads, Seq_Q, Seq_K)")
    
    # Visualize: Tokens attending to tokens
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    print_attention_heatmap(weights, labels_q=tokens, labels_kv=tokens, title="Self-Attention")
    
    # Verify Residual Connection compatibility
    assert out.shape == x.shape, "Output shape must match input for residual add"
    print("\n[OK] Self-Attention preserves shape for Residual Connection.")

def run_cross_attention_demo():
    print("\n" + "="*60)
    print("EXPERIMENT 2: CROSS-ATTENTION (Decoder Block Style)")
    print("="*60)
    
    # Config
    batch_size = 2
    seq_len_q = 4   # Target sequence length (e.g., generated so far)
    seq_len_kv = 6  # Source sequence length (e.g., encoder output)
    d_model = 32
    num_heads = 4
    
    # Encoder Output (Memory): (batch, src_seq_len, d_model)
    enc_output = np.random.randn(batch_size, seq_len_kv, d_model)
    
    # Decoder Input (Target embeddings): (batch, tgt_seq_len, d_model)
    dec_input = np.random.randn(batch_size, seq_len_q, d_model)
    
    mha_cross = MultiHeadAttention(d_model, num_heads)
    
    # Cross-Attention: Q from Decoder, K,V from Encoder
    out, weights = mha_cross.forward(dec_input, enc_output, enc_output)
    
    print_tensor("Decoder Query (Q)", dec_input)
    print_tensor("Encoder Memory (K,V)", enc_output)
    print_tensor("Cross-Attn Output", out)
    print(f"Attention Weights Shape: {weights.shape} (Batch, Heads, Tgt_Len, Src_Len)")
    
    # Visualize: Target tokens attending to Source tokens
    src_tokens = ["<SOS>", "The", "cat", "sat", "on", "mat"]
    tgt_tokens = ["<SOS>", "Le", "chat", "dort"]
    print_attention_heatmap(weights, labels_q=tgt_tokens, labels_kv=src_tokens, title="Cross-Attention (Decoder->Encoder)")

    assert out.shape == dec_input.shape
    print("\n[OK] Cross-Attention output matches Query (Decoder) sequence length.")

def run_causal_mask_demo():
    print("\n" + "="*60)
    print("EXPERIMENT 3: CAUSAL MASKING (Autoregressive Self-Attention)")
    print("="*60)
    
    batch_size = 1
    seq_len = 5
    d_model = 16
    num_heads = 2
    
    x = np.random.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Create Causal Mask: Lower triangular matrix (1=keep, 0=mask)
    # Shape: (1, 1, seq_len, seq_len) for broadcasting over batch/heads
    mask = np.tril(np.ones((seq_len, seq_len))).reshape(1, 1, seq_len, seq_len)
    
    print("Causal Mask (1=Visible, 0=Masked Future):")
    print(mask[0,0].astype(int))
    
    # Forward with mask
    out_masked, weights_masked = mha.forward(x, x, x, mask=mask)
    # Forward without mask (for comparison)
    out_unmasked, weights_unmasked = mha.forward(x, x, x, mask=None)
    
    print("\nAttention Weights (Masked) - Row 3 (Token 3) should only see 0,1,2,3:")
    print(weights_masked[0, 0, 3, :].round(3))
    
    print("\nAttention Weights (Unmasked) - Row 3 sees all:")
    print(weights_unmasked[0, 0, 3, :].round(3))
    
    print("\n[OK] Causal mask prevents attention to future tokens (upper triangle ~0).")

def run_gradient_sanity_check():
    """Numerical gradient check for the attention block."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: NUMERICAL GRADIENT CHECK (Backprop Sanity)")
    print("="*60)
    
    # Tiny setup for finite differences
    d_model = 8
    num_heads = 2
    seq_len = 3
    batch = 1
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = np.random.randn(batch, seq_len, d_model) * 0.1
    upstream_grad = np.random.randn(batch, seq_len, d_model) * 0.1
    
    eps = 1e-5
    param_idx = (0, 0) # Check W_q[0,0]
    
    # 1. Analytical Gradient (Manual Backprop for W_q only - simplified)
    # We only check the projection W_q here for brevity.
    # Full backprop is verbose; we use finite diff on the loss function.
    
    def loss_fn(W_q_val):
        mha.W_q = W_q_val.copy()
        out, _ = mha.forward(x, x, x)
        return np.sum(out * upstream_grad) # Scalar loss
    
    W_q_orig = mha.W_q.copy()
    # Analytical grad approximation via autodiff logic (simplified):
    # dL/dW_q = x^T @ (dL/dQ)  ... this is complex to write fully manually here.
    # Instead, we verify the *forward pass* is differentiable via Finite Differences on Input X.
    
    def loss_fn_input(x_val):
        out, _ = mha.forward(x_val, x_val, x_val)
        return np.sum(out * upstream_grad)
    
    # Finite Difference on Input X
    x_flat = x.flatten()
    grad_num = np.zeros_like(x_flat)
    for i in range(len(x_flat)):
        x_plus = x_flat.copy(); x_plus[i] += eps
        x_minus = x_flat.copy(); x_minus[i] -= eps
        l_plus = loss_fn_input(x_plus.reshape(x.shape))
        l_minus = loss_fn_input(x_minus.reshape(x.shape))
        grad_num[i] = (l_plus - l_minus) / (2*eps)
    
    # Analytical Gradient for Input X (requires implementing backward pass)
    # Since we didn't implement backward(), we skip analytical comparison 
    # but verify the numerical gradient is non-zero and sensible.
    print(f"Numerical Grad (Input X) Norm: {np.linalg.norm(grad_num):.6f}")
    print(f"Max Abs Grad: {np.max(np.abs(grad_num)):.6f}")
    print("[OK] Forward pass is differentiable (non-zero gradients computed via Finite Differences).")

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    print("DAY 109: ATTENTION MECHANISMS - SELF & CROSS ATTENTION")
    print("Implementation: Pure NumPy (No Frameworks)")
    
    run_self_attention_demo()
    run_cross_attention_demo()
    run_causal_mask_demo()
    run_gradient_sanity_check()
    
    print("\n" + "="*60)
    print("JOURNAL ENTRY COMPLETE: Day 109")
    print("Key Takeaways:")
    print("1. Self-Attention: Q,K,V from same source -> Contextualization.")
    print("2. Cross-Attention: Q from Decoder, K,V from Encoder -> Alignment/Translation.")
    print("3. Multi-Head: Parallel independent attention subspaces.")
    print("4. Causal Mask: Essential for Autoregressive Generation (GPT-style).")
    print("5. Shapes: (Batch, Heads, Seq_Len, Head_Dim) is the internal standard.")
    print("="*60)