import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import matplotlib.pyplot as plt

torch.manual_seed(113)
random.seed(113)

# ============================================================
# 1. Tiny character-level dataset
# ============================================================
text = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles "
    "And by opposing end them."
) * 20  # repeat for more training signal

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 64
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# 2. Minimal GPT components
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, att

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        a, attn_weights = self.attn(self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=64, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence too long"
        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, attn_maps

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================
# 3. Training loop
# ============================================================
model = NanoGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
max_iters = 500
eval_interval = 50
eval_iters = 20
train_losses, val_losses = [], []

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"Training on {device} | vocab={vocab_size} | params={sum(p.numel() for p in model.parameters()):,}")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        print(f"iter {iter:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss, _ = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# ============================================================
# 4. Generation & visualization
# ============================================================
context = torch.tensor([encode("To be")], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
print("\n--- Generated sample ---")
print(decode(generated[0].tolist()))

# Plot loss curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
steps = list(range(0, max_iters, eval_interval))
plt.plot(steps, train_losses, label='train')
plt.plot(steps, val_losses, label='val')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualize attention from last layer, first head, first batch item
model.eval()
with torch.no_grad():
    xb, _ = get_batch('val')
    _, _, attn_maps = model(xb)
    attn = attn_maps[-1][0, 0].cpu()  # [T, T]

plt.subplot(1, 2, 2)
im = plt.imshow(attn, cmap='Blues', aspect='auto')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title('Causal Attention Map (Layer 4, Head 1)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.tight_layout()
plt.savefig('day113_attention.png', dpi=150)
print("\nSaved attention visualization to day113_attention.png")

# ============================================================
# 5. Autoregressive property demonstration
# ============================================================
print("\n--- Autoregressive verification ---")
print("Generating token-by-token with probability tracking:")

prompt = "To be"
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
for step in range(10):
    with torch.no_grad():
        logits, _, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / 0.8
        probs = F.softmax(logits, dim=-1)
        topk = torch.topk(probs, 5)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)
        print(f"Step {step+1}: '{decode(next_token[0].tolist())}' | "
              f"Top-5: {[(decode([i.item()]), f'{p.item():.3f}') for i, p in zip(topk.indices[0], topk.values[0])]}")

print(f"\nFinal: '{decode(idx[0].tolist())}'")