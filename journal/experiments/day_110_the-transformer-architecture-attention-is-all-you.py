import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

# ============================================================
# Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 32          # 0: PAD, 1: SOS, 2: EOS, 3-31: tokens
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
D_FF = 512
DROPOUT = 0.1
MAX_LEN = 64
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 10
SEQ_LEN = 16             # Length of random sequences to copy

# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ============================================================
# Transformer Blocks (From Scratch)
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
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
        batch_size = q.size(0)
        # Linear projections & split heads: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v) # (batch, n_heads, seq, d_k)

        # Concat heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.w_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pre-Norm architecture (more stable)
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
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

    def forward(self, x, memory, tgt_mask, src_mask):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, tgt_mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x2, memory, memory, src_mask))
        x2 = self.norm3(x)
        x = x + self.dropout(self.ff(x2))
        return x

# ============================================================
# Full Transformer Model
# ============================================================
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        
        self.norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.norm(x)

    def decode(self, tgt, memory, tgt_mask, src_mask):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(out)

# ============================================================
# Data: Copy Task (Input: SOS + Seq + EOS -> Target: SOS + Seq + EOS)
# ============================================================
PAD, SOS, EOS = 0, 1, 2

def generate_batch(batch_size, seq_len, vocab_size):
    # Random tokens from 3 to vocab_size-1
    data = torch.randint(3, vocab_size, (batch_size, seq_len), device=DEVICE)
    # Source: SOS + data + EOS + PAD...
    src = torch.cat([torch.full((batch_size, 1), SOS, device=DEVICE), data, torch.full((batch_size, 1), EOS, device=DEVICE)], dim=1)
    # Target: SOS + data + EOS (Teacher forcing input)
    tgt_in = torch.cat([torch.full((batch_size, 1), SOS, device=DEVICE), data], dim=1)
    # Target Output: data + EOS
    tgt_out = torch.cat([data, torch.full((batch_size, 1), EOS, device=DEVICE)], dim=1)
    return src, tgt_in, tgt_out

def make_masks(src, tgt):
    # Src mask: (batch, 1, 1, src_len) - True for non-pad
    src_mask = (src != PAD).unsqueeze(1).unsqueeze(2)
    # Tgt mask: (batch, 1, tgt_len, tgt_len) - Causal + Padding
    tgt_len = tgt.size(1)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=DEVICE)).bool()
    pad_mask = (tgt != PAD).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, tgt_len)
    tgt_mask = causal_mask & pad_mask
    return src_mask, tgt_mask

# ============================================================
# Training Loop
# ============================================================
def run_experiment():
    print(f"Device: {DEVICE}")
    model = Transformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, MAX_LEN).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        # Simulate ~1000 batches per epoch
        for _ in range(100):
            src, tgt_in, tgt_out = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
            src_mask, tgt_mask = make_masks(src, tgt_in)
            
            optimizer.zero_grad()
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt_out.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / 100
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f}")

    # ============================================================
    # Inference Demo (Greedy Decoding)
    # ============================================================
    print("\n--- Inference Demo ---")
    model.eval()
    with torch.no_grad():
        # Create a specific test sequence
        test_seq = torch.tensor([[3, 5, 7, 9, 11, 13]], device=DEVICE) # batch=1
        src = torch.cat([torch.tensor([[SOS]], device=DEVICE), test_seq, torch.tensor([[EOS]], device=DEVICE)], dim=1)
        src_mask = (src != PAD).unsqueeze(1).unsqueeze(2)
        
        memory = model.encode(src, src_mask)
        
        # Start decoding with SOS
        ys = torch.tensor([[SOS]], device=DEVICE)
        print(f"Input:  {test_seq.squeeze().tolist()}")
        
        for i in range(SEQ_LEN + 2): # Max length
            tgt_mask = torch.tril(torch.ones(ys.size(1), ys.size(1), device=DEVICE)).bool().unsqueeze(0).unsqueeze(0)
            out = model.decode(ys, memory, tgt_mask, src_mask)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            if next_word == EOS:
                print(f"Output: {ys.squeeze().tolist()[1:]} (EOS reached)")
                break
            ys = torch.cat([ys, torch.tensor([[next_word]], device=DEVICE)], dim=1)
        else:
            print(f"Output: {ys.squeeze().tolist()[1:]} (Max len reached)")

if __name__ == "__main__":
    run_experiment()