import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys

# ============================================================
# Day 31: GPT & Autoregressive Language Modeling Mini-Experiment
# ============================================================

# --- Configuration ---
VOCAB_SIZE = 64          # Character-level vocab (subset of ASCII)
D_MODEL = 128            # Embedding dimension
N_HEADS = 4              # Attention heads
N_LAYERS = 4             # Transformer blocks
D_FF = 512               # Feed-forward dimension
MAX_SEQ_LEN = 128        # Context window
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

# --- Tiny Dataset: Character-level Shakespeare-ish ---
RAW_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die—to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep, perchance to dream—ay, there's the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "When we have shuffled off this mortal coil,\n"
    "Must give us pause—there's the respect\n"
    "That makes calamity of so long life.\n"
) * 20  # Repeat to have enough training data

# Build vocabulary
CHARS = sorted(list(set(RAW_TEXT)))
VOCAB_SIZE = len(CHARS)
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}
ENCODE = lambda s: [STOI[c] for c in s]
DECODE = lambda l: ''.join([ITOS[i] for i in l])

DATA = torch.tensor(ENCODE(RAW_TEXT), dtype=torch.long)
N = int(0.9 * len(DATA))
TRAIN_DATA = DATA[:N]
VAL_DATA = DATA[N:]

print(f"Vocab size: {VOCAB_SIZE}")
print(f"Train tokens: {len(TRAIN_DATA)}, Val tokens: {len(VAL_DATA)}")
print(f"Device: {DEVICE}")

# --- Data Loader ---
def get_batch(split):
    data = TRAIN_DATA if split == 'train' else VAL_DATA
    ix = torch.randint(len(data) - MAX_SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i:i+MAX_SEQ_LEN] for i in ix])
    y = torch.stack([data[i+1:i+MAX_SEQ_LEN+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- Model Components ---
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert D_MODEL % N_HEADS == 0
        self.c_attn = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.c_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).view(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(D_MODEL, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(D_MODEL, D_FF, bias=False)
        self.c_proj = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, D_MODEL))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= MAX_SEQ_LEN, "Sequence too long"
        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -MAX_SEQ_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Training ---
model = GPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("\n--- Training ---")
for epoch in range(EPOCHS):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        losses = estimate_loss()
        print(f"Epoch {epoch:3d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

# --- Generation Demo ---
print("\n--- Generation ---")
context = torch.tensor([ENCODE("To be")], dtype=torch.long, device=DEVICE)
print(f"Prompt: \"{DECODE(context[0].tolist())}\"")

for temp in [0.7, 1.0, 1.2]:
    out = model.generate(context, max_new_tokens=200, temperature=temp, top_k=40)
    print(f"\n[temp={temp}] {DECODE(out[0].tolist())}")

# --- Autoregressive Property Verification ---
print("\n--- Autoregressive Verification ---")
print("Checking that position t only depends on positions <= t...")

model.eval()
with torch.no_grad():
    # Create two sequences that differ only at position 5
    seq1 = torch.randint(0, VOCAB_SIZE, (1, 10), device=DEVICE)
    seq2 = seq1.clone()
    seq2[0, 5] = (seq2[0, 5] + 1) % VOCAB_SIZE  # Change token at pos 5

    logits1, _ = model(seq1)
    logits2, _ = model(seq2)

    # Positions 0-4 should be identical (causal)
    # Position 5+ should differ
    diff = (logits1 - logits2).abs().max(dim=-1).values
    print(f"Max logit diff at positions 0-4 (should be ~0): {diff[0, :5].max().item():.6f}")
    print(f"Max logit diff at positions 5-9 (should be >0): {diff[0, 5:].max().item():.6f}")

    # Verify causal mask in attention
    attn_module = model.blocks[0].attn
    B, T = 1, 8
    dummy = torch.randn(B, T, D_MODEL, device=DEVICE)
    # Hook to capture attention weights
    attn_weights = []
    def hook(module, inp, out):
        # We can't easily get attn weights without modifying forward
        pass
    # Instead, verify mask shape
    print(f"Causal mask shape: {attn_module.mask.shape}")
    print(f"Mask[0,0,:,:] (lower triangular):\n{attn_module.mask[0,0,:8,:8].int()}")

print("\n--- Experiment Complete ---")