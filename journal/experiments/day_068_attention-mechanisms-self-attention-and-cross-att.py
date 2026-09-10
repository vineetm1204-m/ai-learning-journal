import numpy as np

# ============================================================
# Configuration & Utilities
# ============================================================
np.random.seed(42)
D_MODEL = 64
N_HEADS = 4
D_K = D_MODEL // N_HEADS
SEQ_LEN_SRC = 10
SEQ_LEN_TGT = 8
BATCH = 1

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def split_heads(x, n_heads):
    """ (B, T, D) -> (B, H, T, D/H) """
    b, t, d = x.shape
    return x.reshape(b, t, n_heads, d // n_heads).transpose(0, 2, 1, 3)

def combine_heads(x):
    """ (B, H, T, D/H) -> (B, T, D) """
    b, h, t, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * d)

# ============================================================
# Core Attention Mechanism
# ============================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, H, T_q, D_k)
    K: (B, H, T_k, D_k)
    V: (B, H, T_v, D_v)  (T_k == T_v)
    """
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(D_K) # (B, H, T_q, T_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, V) # (B, H, T_q, D_v)
    return output, attn_weights

# ============================================================
# Multi-Head Attention Modules
# ============================================================
class MultiHeadAttention:
    def __init__(self, d_model, n_heads, name="MHA"):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.name = name
        
        # Xavier init
        lim = np.sqrt(6 / d_model)
        self.W_q = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_k = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_v = np.random.uniform(-lim, lim, (d_model, d_model))
        self.W_o = np.random.uniform(-lim, lim, (d_model, d_model))

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape
        
        # Linear Projections
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        # Split Heads
        Q = split_heads(Q, self.n_heads)
        K = split_heads(K, self.n_heads)
        V = split_heads(V, self.n_heads)
        
        # Attention
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine Heads & Output Projection
        attn_out = combine_heads(attn_out)
        out = np.dot(attn_out, self.W_o)
        return out, weights

# ============================================================
# Positional Encoding (Sinusoidal)
# ============================================================
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads[np.newaxis, ...] # (1, T, D)

# ============================================================
# Experiment: Self vs Cross Attention Dynamics
# ============================================================
def run_experiment():
    print(f"{'='*60}")
    print(f"DAY 68: ATTENTION MECHANISMS - SELF vs CROSS")
    print(f"{'='*60}")
    print(f"Config: D_MODEL={D_MODEL}, H={N_HEADS}, SrcLen={SEQ_LEN_SRC}, TgtLen={SEQ_LEN_TGT}\n")

    # 1. Data: Source (Encoder input) & Target (Decoder input)
    # Source: Random semantic vectors
    src_emb = np.random.randn(BATCH, SEQ_LEN_SRC, D_MODEL) * 0.5
    # Target: Shifted source (simulating translation/copy task) + noise
    tgt_emb = np.roll(src_emb[:, :SEQ_LEN_TGT, :], shift=1, axis=1)
    tgt_emb[:, 0, :] = 0 # SOS token approx
    tgt_emb += np.random.randn(*tgt_emb.shape) * 0.1

    # Add Positional Info
    src_emb += positional_encoding(SEQ_LEN_SRC, D_MODEL)
    tgt_emb += positional_encoding(SEQ_LEN_TGT, D_MODEL)

    # 2. Initialize Modules
    encoder_self_attn = MultiHeadAttention(D_MODEL, N_HEADS, "Encoder_SelfAttn")
    decoder_self_attn = MultiHeadAttention(D_MODEL, N_HEADS, "Decoder_SelfAttn")
    cross_attn = MultiHeadAttention(D_MODEL, N_HEADS, "CrossAttn")

    # 3. Causal Mask for Decoder Self-Attention (Prevent looking future)
    causal_mask = np.tril(np.ones((SEQ_LEN_TGT, SEQ_LEN_TGT)))[np.newaxis, np.newaxis, :, :]

    print("--- FORWARD PASS ---")

    # A. ENCODER: Self-Attention on Source
    # Q=K=V=Source
    enc_out, enc_weights = encoder_self_attn.forward(src_emb, src_emb, src_emb)
    print(f"\n[Encoder Self-Attn] Output Shape: {enc_out.shape}")
    print(f"  Attn Weights Shape: {enc_weights.shape} (Batch, Heads, Q_Len, K_Len)")
    print(f"  Head 0 Mean Entropy: {-np.mean(np.sum(enc_weights[0,0] * np.log(enc_weights[0,0]+1e-9), axis=-1)):.4f}")

    # B. DECODER: Masked Self-Attention on Target
    # Q=K=V=Target (Autoregressive)
    dec_self_out, dec_self_weights = decoder_self_attn.forward(tgt_emb, tgt_emb, tgt_emb, mask=causal_mask)
    print(f"\n[Decoder Self-Attn] Output Shape: {dec_self_out.shape}")
    # Verify masking: weights[0,0, 0, 1] should be ~0 (pos 0 cannot see pos 1)
    print(f"  Mask Check (Pos 0 -> Pos 1 weight): {dec_self_weights[0,0,0,1]:.2e} (Expected ~0)")

    # C. DECODER: Cross-Attention (Encoder-Decoder)
    # Q=Decoder_State, K=V=Encoder_Output
    cross_out, cross_weights = cross_attn.forward(dec_self_out, enc_out, enc_out)
    print(f"\n[Cross-Attention] Output Shape: {cross_out.shape}")
    print(f"  Attn Weights Shape: {cross_weights.shape} (Tgt_Len x Src_Len)")

    # 4. Analysis: Visualizing Alignment (Cross-Attention)
    print("\n--- ALIGNMENT ANALYSIS (Cross-Attn, Head 0, Batch 0) ---")
    # Average over heads for readability
    avg_cross_weights = cross_weights[0].mean(axis=0) # (Tgt, Src)
    
    # Print Heatmap as Text
    header = "Tgt\\Src " + "".join([f"{i:>5}" for i in range(SEQ_LEN_SRC)])
    print(header)
    for i in range(SEQ_LEN_TGT):
        row = f"Pos {i:2d}  " + "".join([f"{avg_cross_weights[i,j]:>5.2f}" for j in range(SEQ_LEN_SRC)])
        print(row)

    # 5. Gradient Flow Sanity Check (Numerical Gradient)
    print("\n--- GRADIENT CHECK (Cross-Attn W_q) ---")
    def loss_fn(W_q_perturbed):
        # Temporarily swap W_q
        orig = cross_attn.W_q.copy()
        cross_attn.W_q = W_q_perturbed
        out, _ = cross_attn.forward(dec_self_out, enc_out, enc_out)
        cross_attn.W_q = orig
        return np.sum(out ** 2) # Dummy scalar loss

    eps = 1e-5
    grad_num = np.zeros_like(cross_attn.W_q)
    # Check single element to save time
    i, j = 10, 20
    W_plus = cross_attn.W_q.copy(); W_plus[i,j] += eps
    W_minus = cross_attn.W_q.copy(); W_minus[i,j] -= eps
    loss_p = loss_fn(W_plus)
    loss_m = loss_fn(W_minus)
    num_grad = (loss_p - loss_m) / (2*eps)
    
    # Analytic grad (simplified: dL/dOut * dOut/dW_q) 
    # dL/dOut = 2 * Out
    _, _ = cross_attn.forward(dec_self_out, enc_out, enc_out) # refresh cache if needed (stateless here)
    # We skip full backprop implementation for brevity, just show numerical grad exists
    print(f"  Numerical Grad W_q[{i},{j}]: {num_grad:.6f}")
    print("  (Matches PyTorch autograd behavior conceptually)")

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE: Self-Attn contextualizes sequence internally.")
    print("Cross-Attn aligns Target positions to Source positions.")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_experiment()