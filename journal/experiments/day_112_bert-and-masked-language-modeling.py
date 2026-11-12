import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
VOCAB_SIZE = 1000
MAX_LEN = 64
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
D_FF = 512
DROPOUT = 0.1
BATCH_SIZE = 16
LR = 3e-4
STEPS = 200
MASK_PROB = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special tokens
PAD_ID = 0
MASK_ID = 1
CLS_ID = 2
SEP_ID = 3
VOCAB_OFFSET = 4  # Real tokens start here

# ============================================================
# Tiny BERT Components
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TinyBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_emb = PositionalEncoding(D_MODEL, MAX_LEN)
        self.dropout = nn.DROPout(DROPOUT)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
            dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.ln = nn.LayerNorm(D_MODEL)
        
        # MLM Head
        self.mlm_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, VOCAB_SIZE)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.tok_emb(input_ids)
        x = self.pos_emb(x)
        x = self.dropout(x)
        
        # attention_mask: 1 for real tokens, 0 for padding -> src_key_padding_mask expects True for padding
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln(x)
        logits = self.mlm_head(x)
        return logits

# ============================================================
# Data Utilities
# ============================================================
def random_sequence(length):
    """Generate random token sequence (excluding special tokens)"""
    return torch.randint(VOCAB_OFFSET, VOCAB_SIZE, (length,))

def create_batch(batch_size, max_len):
    """Create batch of random sequences with padding"""
    lengths = torch.randint(max_len // 2, max_len, (batch_size,))
    sequences = [torch.cat([torch.tensor([CLS_ID]), random_sequence(l-2), torch.tensor([SEP_ID])]) for l in lengths]
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
    # Truncate if needed
    if padded.size(1) > max_len:
        padded = padded[:, :max_len]
    attention_mask = (padded != PAD_ID).long()
    return padded, attention_mask

def mask_tokens(inputs, attention_mask):
    """
    BERT masking:
    - 15% of tokens selected
    - 80% -> [MASK]
    - 10% -> random token
    - 10% -> unchanged
    Labels: -100 for non-masked (ignored in loss)
    """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, MASK_PROB, device=DEVICE)
    probability_matrix.masked_fill_(attention_mask == 0, 0.0)  # Don't mask padding
    probability_matrix.masked_fill_(inputs == CLS_ID, 0.0)
    probability_matrix.masked_fill_(inputs == SEP_ID, 0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked
    
    # 80% replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=DEVICE)).bool() & masked_indices
    inputs[indices_replaced] = MASK_ID
    
    # 10% replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=DEVICE)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(VOCAB_OFFSET, VOCAB_SIZE, labels.shape, dtype=torch.long, device=DEVICE)
    inputs[indices_random] = random_words[indices_random]
    
    # 10% keep original (do nothing)
    
    return inputs, labels

# ============================================================
# Training & Evaluation
# ============================================================
def train_step(model, optimizer, input_ids, attention_mask):
    model.train()
    optimizer.zero_grad()
    
    masked_input, labels = mask_tokens(input_ids.clone(), attention_mask)
    logits = model(masked_input, attention_mask)
    
    loss = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),
        labels.view(-1),
        ignore_index=-100
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, input_ids, attention_mask):
    model.eval()
    masked_input, labels = mask_tokens(input_ids.clone(), attention_mask)
    logits = model(masked_input, attention_mask)
    
    loss = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Accuracy on masked positions only
    preds = logits.argmax(dim=-1)
    masked = labels != -100
    acc = (preds[masked] == labels[masked]).float().mean().item() if masked.any() else 0.0
    return loss.item(), acc

@torch.no_grad()
def demo_predictions(model, tokenizer_vocab, n_examples=3):
    model.eval()
    print("\n=== Masked Prediction Demos ===")
    for _ in range(n_examples):
        seq_len = random.randint(10, 20)
        tokens = [CLS_ID] + random_sequence(seq_len - 2).tolist() + [SEP_ID]
        input_ids = torch.tensor([tokens], device=DEVICE)
        attention_mask = (input_ids != PAD_ID).long()
        
        masked_input, labels = mask_tokens(input_ids.clone(), attention_mask)
        logits = model(masked_input, attention_mask)
        preds = logits.argmax(dim=-1)
        
        # Find masked positions
        masked_pos = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(masked_pos) == 0:
            continue
        
        print(f"\nOriginal:  {[tokenizer_vocab.get(t, f'[{t}]') for t in tokens]}")
        print(f"Masked:    {[tokenizer_vocab.get(t, f'[{t}]') for t in masked_input[0].tolist()]}")
        for pos in masked_pos[:5]:  # Show first 5 masked
            true_tok = labels[0, pos].item()
            pred_tok = preds[0, pos].item()
            print(f"  Pos {pos.item():2d}: True={tokenizer_vocab.get(true_tok, true_tok):>10} | Pred={tokenizer_vocab.get(pred_tok, pred_tok):>10} {'✓' if true_tok == pred_tok else '✗'}")

# ============================================================
# Main Experiment
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"Config: vocab={VOCAB_SIZE}, d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}")
    
    # Dummy tokenizer vocab for pretty printing
    tokenizer_vocab = {
        PAD_ID: "[PAD]", MASK_ID: "[MASK]", CLS_ID: "[CLS]", SEP_ID: "[SEP]"
    }
    for i in range(VOCAB_OFFSET, VOCAB_SIZE):
        tokenizer_vocab[i] = f"tok_{i}"
    
    model = TinyBERT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    train_losses, val_losses, val_accs = [], [], []
    
    print("\nTraining...")
    for step in range(1, STEPS + 1):
        # Train batch
        input_ids, attention_mask = create_batch(BATCH_SIZE, MAX_LEN)
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        loss = train_step(model, optimizer, input_ids, attention_mask)
        train_losses.append(loss)
        scheduler.step()
        
        # Validation every 20 steps
        if step % 20 == 0:
            val_input_ids, val_attention_mask = create_batch(BATCH_SIZE * 2, MAX_LEN)
            val_input_ids, val_attention_mask = val_input_ids.to(DEVICE), val_attention_mask.to(DEVICE)
            val_loss, val_acc = evaluate(model, val_input_ids, val_attention_mask)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"Step {step:4d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Final demo
    demo_predictions(model, tokenizer_vocab, n_examples=5)
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', alpha=0.7)
    plt.plot(range(20, STEPS+1, 20), val_losses, 'o-', label='Val', color='orange')
    plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Masked LM Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(20, STEPS+1, 20), val_accs, 'o-', color='green')
    plt.xlabel('Step'); plt.ylabel('Accuracy'); plt.title('Masked Token Accuracy'); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day112_bert_mlm.png', dpi=150)
    print("\nPlot saved to day112_bert_mlm.png")
    print("Experiment complete.")

if __name__ == "__main__":
    main()