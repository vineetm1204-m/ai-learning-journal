import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class SelfAttention:
    def __init__(self, d_model, d_k=None, d_v=None):
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        
        self.W_q = np.random.randn(d_model, self.d_k) * 0.02
        self.W_k = np.random.randn(d_model, self.d_k) * 0.02
        self.W_v = np.random.randn(d_model, self.d_v) * 0.02
        self.W_o = np.random.randn(self.d_v, d_model) * 0.02
    
    def forward(self, X, mask=None):
        """
        X: (seq_len, d_model)
        Returns: output (seq_len, d_model), attention_weights (seq_len, seq_len)
        """
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        
        scores = Q @ K.T / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = softmax(scores, axis=-1)
        output = attn_weights @ V
        output = output @ self.W_o
        
        return output, attn_weights

class CrossAttention:
    def __init__(self, d_model, d_k=None, d_v=None):
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        
        self.W_q = np.random.randn(d_model, self.d_k) * 0.02
        self.W_k = np.random.randn(d_model, self.d_k) * 0.02
        self.W_v = np.random.randn(d_model, self.d_v) * 0.02
        self.W_o = np.random.randn(self.d_v, d_model) * 0.02
    
    def forward(self, query, key_value, mask=None):
        """
        query: (seq_len_q, d_model) - decoder side
        key_value: (seq_len_kv, d_model) - encoder side
        Returns: output (seq_len_q, d_model), attention_weights (seq_len_q, seq_len_kv)
        """
        Q = query @ self.W_q
        K = key_value @ self.W_k
        V = key_value @ self.W_v
        
        scores = Q @ K.T / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = softmax(scores, axis=-1)
        output = attn_weights @ V
        output = output @ self.W_o
        
        return output, attn_weights

def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    return mask

def visualize_attention(weights, title, x_labels=None, y_labels=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            ax.text(j, i, f'{weights[i, j]:.2f}', ha='center', va='center', 
                   color='white' if weights[i, j] > 0.5 else 'black', fontsize=9)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax

print("=" * 60)
print("DAY 27: ATTENTION MECHANISMS - SELF & CROSS ATTENTION")
print("=" * 60)

d_model = 64
seq_len = 6

tokens = ["The", "cat", "sat", "on", "the", "mat"]
X = np.random.randn(seq_len, d_model) * 0.1

for i, token in enumerate(tokens):
    X[i] += np.eye(d_model)[i % d_model] * 2.0

print(f"\nInput sequence: {tokens}")
print(f"Input shape: {X.shape}")

print("\n" + "-" * 60)
print("1. SELF-ATTENTION (Encoder-style)")
print("-" * 60)

self_attn = SelfAttention(d_model)
output_sa, weights_sa = self_attn.forward(X)

print(f"Output shape: {output_sa.shape}")
print(f"Attention weights shape: {weights_sa.shape}")
print(f"\nAttention weights (row i attends to column j):")
print(weights_sa.round(3))

print("\nRow sums (should be ~1.0):", weights_sa.sum(axis=1).round(4))

print("\n" + "-" * 60)
print("2. SELF-ATTENTION WITH CAUSAL MASK (Decoder-style)")
print("-" * 60)

causal_mask = create_causal_mask(seq_len)
output_sa_masked, weights_sa_masked = self_attn.forward(X, mask=causal_mask)

print("Causal mask applied (upper triangular = -inf):")
print(causal_mask.round(1))
print("\nMasked attention weights:")
print(weights_sa_masked.round(3))

print("\n" + "-" * 60)
print("3. CROSS-ATTENTION (Encoder-Decoder)")
print("-" * 60)

encoder_seq = ["<ENC>", "The", "cat", "is", "black", "<EOS>"]
decoder_seq = ["<DEC>", "Le", "chat", "est", "noir", "<EOS>"]

enc_len, dec_len = len(encoder_seq), len(decoder_seq)
encoder_out = np.random.randn(enc_len, d_model) * 0.1
decoder_in = np.random.randn(dec_len, d_model) * 0.1

for i in range(enc_len):
    encoder_out[i] += np.eye(d_model)[i % d_model] * 2.0
for i in range(dec_len):
    decoder_in[i] += np.eye(d_model)[(i + 10) % d_model] * 2.0

cross_attn = CrossAttention(d_model)
output_ca, weights_ca = cross_attn.forward(decoder_in, encoder_out)

print(f"Encoder sequence: {encoder_seq}")
print(f"Decoder sequence: {decoder_seq}")
print(f"Encoder output shape: {encoder_out.shape}")
print(f"Decoder input shape: {decoder_in.shape}")
print(f"Cross-attention output shape: {output_ca.shape}")
print(f"Cross-attention weights shape: {weights_ca.shape}")
print(f"\nCross-attention weights (decoder rows attend to encoder columns):")
print(weights_ca.round(3))

print("\n" + "-" * 60)
print("4. MULTI-HEAD ATTENTION DEMONSTRATION")
print("-" * 60)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, X, mask=None):
        seq_len = X.shape[0]
        
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = softmax(scores, axis=-1)
        output = np.matmul(attn_weights, V)
        
        output = output.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        output = output @ self.W_o
        
        return output, attn_weights

mha = MultiHeadAttention(d_model, num_heads=4)
output_mha, weights_mha = mha.forward(X)

print(f"Multi-head attention output shape: {output_mha.shape}")
print(f"Attention weights per head shape: {weights_mha.shape}")
print(f"\nHead 0 weights:")
print(weights_mha[0].round(3))
print(f"\nHead 1 weights:")
print(weights_mha[1].round(3))
print(f"\nHead 2 weights:")
print(weights_mha[2].round(3))
print(f"\nHead 3 weights:")
print(weights_mha[3].round(3))

print("\n" + "-" * 60)
print("5. VISUALIZATION")
print("-" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

visualize_attention(weights_sa, "Self-Attention (Encoder)", 
                    x_labels=tokens, y_labels=tokens, ax=axes[0, 0])

visualize_attention(weights_sa_masked, "Self-Attention (Causal/Decoder)", 
                    x_labels=tokens, y_labels=tokens, ax=axes[0, 1])

visualize_attention(weights_ca, "Cross-Attention (Decoder→Encoder)", 
                    x_labels=encoder_seq, y_labels=decoder_seq, ax=axes[0, 2])

for h in range(4):
    ax = axes[1, h] if h < 2 else axes[1, h-2] if h < 4 else None
    if h < 2:
        visualize_attention(weights_mha[h], f"Head {h}", 
                            x_labels=tokens, y_labels=tokens, ax=axes[1, h])
    elif h < 4:
        visualize_attention(weights_mha[h], f"Head {h}", 
                            x_labels=tokens, y_labels=tokens, ax=axes[1, h-2])

axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, 'Heads 2 & 3\n(see console)', ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'attention_visualization.png'")

print("\n" + "=" * 60)
print("KEY INSIGHTS:")
print("=" * 60)
print("""
1. SELF-ATTENTION: Each token attends to ALL tokens (including itself).
   - Encoder: Full visibility (bidirectional)
   - Decoder: Causal mask prevents attending to future tokens

2. CROSS-ATTENTION: Decoder queries attend to Encoder keys/values.
   - Query comes from decoder, Key/Value from encoder
   - Enables sequence-to-sequence tasks (translation, summarization)

3. MULTI-HEAD: Parallel attention heads capture different relationships.
   - Head 0 might capture syntactic dependencies
   - Head 1 might capture semantic relationships
   - Head 2 might capture positional patterns
   - Head 3 might capture long-range dependencies

4. SCALED DOT-PRODUCT: Division by sqrt(d_k) prevents gradient vanishing
   for large dimensions by keeping variance stable.

5. SOFTMAX: Converts scores to probability distribution over keys.
""")

print("Experiment complete!")