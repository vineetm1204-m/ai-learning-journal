import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys
from collections import Counter

torch.manual_seed(42)
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────
# 1. Tiny character-level tokenizer
# ──────────────────────────────────────────────
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

# ──────────────────────────────────────────────
# 2. Minimal GPT building blocks
# ──────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 128,
        dropout: float = 0.1,
    ):
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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence longer than block size"
        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


# ──────────────────────────────────────────────
# 3. Data loading (tiny Shakespeare subset)
# ──────────────────────────────────────────────
def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ──────────────────────────────────────────────
# 4. Training loop + evaluation
# ──────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, data_train, data_val, batch_size, block_size, eval_iters=50):
    out = {}
    model.eval()
    for split, data in [("train", data_train), ("val", data_val)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train():
    # ── hyper-parameters ──────────────────────
    batch_size = 32
    block_size = 128
    max_iters = 2000
    eval_interval = 200
    learning_rate = 3e-4
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.1
    # ──────────────────────────────────────────

    # Tiny corpus (≈ 100 KB)
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) < 1000:
        text = text * (1000 // len(text) + 1)  # repeat if tiny

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    for iter_num in range(max_iters):
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size)
            print(f"iter {iter_num:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, batch_size, block_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ── final generation demo ─────────────────
    model.eval()
    context = torch.tensor([tokenizer.encode("ROMEO:")], dtype=torch.long, device=DEVICE)
    out = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=40)
    print("\n=== GENERATION ===")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    # Create a tiny input.txt if missing
    try:
        open("input.txt").close()
    except FileNotFoundError:
        with open("input.txt", "w") as f:
            f.write(
                "ROMEO: But soft, what light through yonder window breaks?\n"
                "JULIET: O Romeo, Romeo, wherefore art thou Romeo?\n"
            ) * 500  # repeat to get ~100 KB
    train()