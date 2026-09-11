import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Hyperparameters
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.1
MAX_LEN = 20
VOCAB_SIZE = 12  # 0=PAD, 1=SOS, 2=EOS, 3-11=tokens
BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-4

# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============================================================
# Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        q = self.w_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_o(out), attn

# ============================================================
# Feed Forward
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# Encoder / Decoder Layers
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out, cross_attn_weights = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)
        return x, cross_attn_weights

# ============================================================
# Full Transformer
# ============================================================
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, T = tgt.shape
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.tril(torch.ones((T, T), device=DEVICE)).bool()
        return pad_mask & sub_mask

    def encode(self, src, src_mask):
        x = self.dropout(self.pos_enc(self.embed(src)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.dropout(self.pos_enc(self.embed(tgt)))
        cross_attns = []
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, enc_out, tgt_mask, src_mask)
            cross_attns.append(cross_attn)
        return self.fc_out(x), cross_attns

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        out, cross_attns = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return out, cross_attns

# ============================================================
# Synthetic Copy Task Dataset
# ============================================================
def generate_batch(batch_size, max_len, vocab_size):
    seq_len = torch.randint(3, max_len, (batch_size,))
    src = torch.randint(3, vocab_size, (batch_size, max_len))
    for i, l in enumerate(seq_len):
        src[i, l:] = 0
    tgt = src.clone()
    tgt_in = torch.cat([torch.full((batch_size, 1), 1), tgt[:, :-1]], dim=1)
    tgt_out = tgt.clone()
    return src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)

# ============================================================
# Training
# ============================================================
model = Transformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print(f"Device: {DEVICE}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

train_losses = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for _ in range(100):
        src, tgt_in, tgt_out = generate_batch(BATCH_SIZE, MAX_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        logits, _ = model(src, tgt_in)
        loss = criterion(logits.view(-1, VOCAB_SIZE), tgt_out.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / 100
    train_losses.append(avg_loss)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

# ============================================================
# Evaluation & Attention Visualization
# ============================================================
model.eval()
with torch.no_grad():
    src, tgt_in, tgt_out = generate_batch(1, MAX_LEN, VOCAB_SIZE)
    logits, cross_attns = model(src, tgt_in)
    pred = logits.argmax(-1).squeeze(0).cpu().numpy()
    src_seq = src.squeeze(0).cpu().numpy()
    tgt_seq = tgt_out.squeeze(0).cpu().numpy()

    print("\n--- Sample Inference ---")
    print(f"Source:      {src_seq}")
    print(f"Target:      {tgt_seq}")
    print(f"Prediction:  {pred}")

    # Plot cross-attention from last decoder layer, head 0
    attn = cross_attns[-1][0, 0].cpu().numpy()  # [T_tgt, T_src]
    T_tgt = (tgt_in != 0).sum().item()
    T_src = (src != 0).sum().item()
    attn = attn[:T_tgt, :T_src]

    plt.figure(figsize=(6, 5))
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Source Position')
    plt.ylabel('Target Position')
    plt.title('Cross-Attention (Decoder Layer 2, Head 1)')
    plt.xticks(range(T_src), [str(x) for x in src_seq[:T_src]])
    plt.yticks(range(T_tgt), [str(x) for x in tgt_seq[:T_tgt]])
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    print("\nAttention heatmap saved to attention_heatmap.png")

# Plot training curve
plt.figure()
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss (Copy Task)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
print("Training curve saved to training_curve.png")