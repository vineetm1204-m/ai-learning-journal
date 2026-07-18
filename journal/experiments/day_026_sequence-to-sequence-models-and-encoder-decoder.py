import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# Day 26: Sequence-to-Sequence Encoder-Decoder Mini-Experiment
# Task: Learn to reverse input strings (character-level)
# ============================================================

# ---------------------------
# Configuration
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 128
EMBEDDING_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
TEACHER_FORCING_RATIO = 0.5
MAX_LENGTH = 20
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

print(f"Device: {DEVICE}")

# ---------------------------
# Vocabulary & Data Generation
# ---------------------------
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 "
VOCAB = {c: i+3 for i, c in enumerate(CHARS)}  # 0,1,2 reserved
VOCAB['<SOS>'] = SOS_TOKEN
VOCAB['<EOS>'] = EOS_TOKEN
VOCAB['<PAD>'] = PAD_TOKEN
IDX2CHAR = {i: c for c, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)

def generate_reverse_pairs(num_samples=5000, max_len=15):
    """Generate (input, target) pairs where target = reversed input"""
    pairs = []
    for _ in range(num_samples):
        length = random.randint(3, max_len)
        src = ''.join(random.choices(CHARS, k=length))
        tgt = src[::-1]  # reversed
        pairs.append((src, tgt))
    return pairs

def encode_sequence(seq, add_sos=False, add_eos=True):
    """Convert string to tensor of token IDs"""
    ids = [VOCAB[c] for c in seq]
    if add_sos:
        ids = [SOS_TOKEN] + ids
    if add_eos:
        ids = ids + [EOS_TOKEN]
    return torch.tensor(ids, dtype=torch.long)

def collate_batch(batch):
    """Pad sequences in a batch"""
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    
    src_padded = torch.full((len(batch), max_src), PAD_TOKEN, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), PAD_TOKEN, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    return src_padded, tgt_padded, src_lens, tgt_lens

# Generate data
train_pairs = generate_reverse_pairs(4000, MAX_LENGTH)
val_pairs = generate_reverse_pairs(500, MAX_LENGTH)

train_data = [(encode_sequence(s), encode_sequence(t, add_sos=True)) for s, t in train_pairs]
val_data = [(encode_sequence(s), encode_sequence(t, add_sos=True)) for s, t in val_pairs]

# ---------------------------
# Model: Encoder-Decoder with Attention
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, 
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, src, src_lens):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, embed_dim]
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens, 
                                                   batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # outputs: [batch, src_len, hidden_size*2] (bidirectional)
        # hidden: [num_layers*2, batch, hidden_size]
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch, dec_hidden_dim] (last decoder hidden)
        # encoder_outputs: [batch, src_len, enc_hidden_dim*2]
        batch_size, src_len, _ = encoder_outputs.shape
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, dec_hidden]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        return torch.softmax(attention, dim=1)  # [batch, src_len]

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_size, dec_hidden_size, num_layers, dropout, attention):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.attention = attention
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(embed_dim + enc_hidden_size*2, dec_hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden_size + enc_hidden_size*2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        # input_token: [batch] (single token)
        # hidden: [num_layers, batch, dec_hidden_size]
        # encoder_outputs: [batch, src_len, enc_hidden*2]
        
        input_token = input_token.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, embed_dim]
        
        # Attention
        # Use last layer hidden state for attention
        query = hidden[-1]  # [batch, dec_hidden_size]
        attn_weights = self.attention(query, encoder_outputs, mask)  # [batch, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, src_len]
        
        # Context vector
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, enc_hidden*2]
        
        # LSTM input: embedded + context
        lstm_input = torch.cat([embedded, context], dim=2)  # [batch, 1, embed+enc*2]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch, 1, dec_hidden_size]
        
        # Prediction
        output = output.squeeze(1)  # [batch, dec_hidden]
        context = context.squeeze(1)  # [batch, enc_hidden*2]
        embedded = embedded.squeeze(1)  # [batch, embed_dim]
        
        prediction = self.fc_out(torch.cat([output, context, embedded], dim=1))
        return prediction, hidden, cell, attn_weights.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def create_mask(self, src):
        return (src != PAD_TOKEN).float()
    
    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        # hidden: [num_layers*2, batch, hidden] -> need to combine bidirectional
        # Combine bidirectional: [num_layers, batch, hidden*2] -> project to dec_hidden
        # For simplicity, we'll just use the forward direction's last layer
        # Actually, let's properly combine: reshape and sum/concat
        hidden = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # [num_layers, batch, hidden*2]
        cell = cell.view(self.encoder.num_layers, 2, batch_size, -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        
        # Project encoder hidden to decoder hidden size if needed
        # For now assume enc_hidden*2 == dec_hidden_size
        
        mask = self.create_mask(src)
        input_token = tgt[:, 0]  # <SOS>
        
        for t in range(1, tgt_len):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = prediction
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1
            
        return outputs
    
    def translate(self, src, src_lens, max_len=MAX_LENGTH):
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            encoder_outputs, hidden, cell = self.encoder(src, src_lens)
            
            hidden = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
            cell = cell.view(self.encoder.num_layers, 2, batch_size, -1)
            cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
            
            mask = self.create_mask(src)
            input_token = torch.full((batch_size,), SOS_TOKEN, dtype=torch.long, device=self.device)
            
            outputs = []
            attentions = []
            
            for _ in range(max_len):
                prediction, hidden, cell, attn = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
                top1 = prediction.argmax(1)
                outputs.append(top1)
                attentions.append(attn)
                input_token = top1
                if (top1 == EOS_TOKEN).all():
                    break
                    
        return torch.stack(outputs, dim=1), torch.stack(attentions, dim=1)

# ---------------------------
# Initialize Model
# ---------------------------
ENC_HIDDEN = HIDDEN_SIZE
DEC_HIDDEN = HIDDEN_SIZE * 2  # Match encoder bidirectional output

attention = Attention(ENC_HIDDEN * 2, DEC_HIDDEN)
encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, ENC_HIDDEN, NUM_LAYERS, DROPOUT).to(DEVICE)
decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, ENC_HIDDEN, DEC_HIDDEN, NUM_LAYERS, DROPOUT, attention).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------
# Training Loop
# ---------------------------
def train_epoch(model, data, optimizer, criterion, clip=1.0):
    model.train()
    total_loss = 0
    random.shuffle(data)
    
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        src, tgt, src_lens, tgt_lens = collate_batch(batch)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(src, src_lens, tgt, TEACHER_FORCING_RATIO)
        # output: [batch, tgt_len, vocab], tgt: [batch, tgt_len]
        loss = criterion(output[:, 1:].reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / (len(data) / BATCH_SIZE)

def evaluate(model, data, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            src, tgt, src_lens, tgt_lens = collate_batch(batch)
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(output[:, 1:].reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            # Accuracy (token-level)
            pred = output[:, 1:].argmax(-1)
            mask = (tgt[:, 1:] != PAD_TOKEN)
            correct += ((pred == tgt[:, 1:]) & mask).sum().item()
            total += mask.sum().item()
            
    return total_loss / (len(data) / BATCH_SIZE), correct / total if total > 0 else 0

# Training
train_losses, val_losses, val_accs = [], [], []

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_data, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_data, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ---------------------------
# Qualitative Evaluation
# -------------------------==
def decode_sequence(ids):
    chars = []
    for idx in ids:
        if idx.item() == EOS_TOKEN:
            break
        if idx.item() not in (SOS_TOKEN, PAD_TOKEN):
            chars.append(IDX2CHAR.get(idx.item(), '?'))
    return ''.join(chars)

print("\n" + "="*60)
print("QUALITATIVE EXAMPLES")
print("="*60)

model.eval()
with torch.no_grad():
    for _ in range(10):
        src_str, tgt_str = random.choice(val_pairs)
        src_tensor = encode_sequence(src_str).unsqueeze(0).to(DEVICE)
        src_len = [len(src_str)]
        
        pred_ids, attn = model.translate(src_tensor, src_len)
        pred_str = decode_sequence(pred_ids[0])
        
        status = "✓" if pred_str == tgt_str else "✗"
        print(f"{status} Input:    '{src_str}'")
        print(f"   Target:  '{tgt_str}'")
        print(f"   Predict: '{pred_str}'")
        print()

# ---------------------------
# Attention Visualization (one example)
# ---------------------------
print("="*60)
print("ATTENTION VISUALIZATION (last example)")
print("="*60)

src_str, tgt_str = val_pairs[0]
src_tensor = encode_sequence(src_str).unsqueeze(0).to(DEVICE)
src_len = [len(src_str)]
pred_ids, attn = model.translate(src_tensor, src_len)
pred_str = decode_sequence(pred_ids[0])

print(f"Input:  '{src_str}'")
print(f"Target: '{tgt_str}'")
print(f"Pred:   '{pred_str}'")
print()

# Print attention matrix
attn_np = attn[0].cpu().numpy()  # [tgt_len, src_len]
src_tokens = list(src_str)
tgt_tokens = list(pred_str)

print("Attention weights (rows=target, cols=source):")
header = "       " + "  ".join(f"{c:>2}" for c in src_tokens)
print(header)
for i, (t_char, row) in enumerate(zip(tgt_tokens, attn_np)):
    row_str = "  ".join(f"{w:.2f}" for w in row[:len(src_tokens)])
    print(f"  {t_char:>2} | {row_str}")

# ---------------------------
# Plot Training Curves
# ---------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('CrossEntropy Loss')
plt.title('Training Curves')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Val Token Accuracy', color='green', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day26_seq2seq_training.png', dpi=150)
print("\nPlot saved to 'day26_seq2seq_training.png'")

# ---------------------------
# Summary
# ---------------------------
print("\n" + "="*60)
print("DAY 26 SUMMARY: Sequence-to-Sequence Encoder-Decoder")
print("="*60)
print(f"""
Architecture:
  - Encoder: {NUM_LAYERS}-layer Bidirectional LSTM (hidden={ENC_HIDDEN})
  - Decoder: {NUM_LAYERS}-layer LSTM with Bahdanau Attention (hidden={DEC_HIDDEN})
  - Teacher Forcing Ratio: {TEACHER_FORCING_RATIO}
  - Vocab Size: {VOCAB_SIZE} (chars + special tokens)

Task: Character-level string reversal
  - Input:  "hello"  -> Target: "olleh"
  - Trained on {len(train_pairs)} synthetic pairs

Key Concepts Demonstrated:
  1. Encoder processes input sequence -> contextual representations
  2. Attention allows decoder to focus on relevant encoder states
  3. Teacher forcing stabilizes training
  4. Autoregressive generation at inference time
  5. Bidirectional encoder captures full context

Final Val Accuracy: {val_accs[-1]:.4f}
""")