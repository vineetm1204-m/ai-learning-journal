import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time

# ============================================================
# Day 108: Sequence-to-Sequence Encoder-Decoder Mini-Experiment
# Task: Learn to reverse input sequences (character-level)
# ============================================================

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 128
EMBEDDING_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 15
TEACHER_FORCING_RATIO = 0.5
MAX_LENGTH = 20
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

print(f"Device: {DEVICE}")

# -----------------------------
# Vocabulary & Data Generation
# -----------------------------
class Vocab:
    def __init__(self):
        self.char2idx = {'<SOS>': SOS_TOKEN, '<EOS>': EOS_TOKEN, '<PAD>': PAD_TOKEN}
        self.idx2char = {SOS_TOKEN: '<SOS>', EOS_TOKEN: '<EOS>', PAD_TOKEN: '<PAD>'}
        self.n_chars = 3
    
    def add_string(self, s):
        for ch in s:
            if ch not in self.char2idx:
                self.char2idx[ch] = self.n_chars
                self.idx2char[self.n_chars] = ch
                self.n_chars += 1
    
    def encode(self, s, max_len=MAX_LENGTH):
        ids = [self.char2idx.get(ch, self.char2idx['<PAD>']) for ch in s[:max_len-1]]
        ids.append(EOS_TOKEN)
        return ids + [PAD_TOKEN] * (max_len - len(ids))
    
    def decode(self, ids):
        chars = []
        for idx in ids:
            if idx == EOS_TOKEN or idx == PAD_TOKEN:
                break
            chars.append(self.idx2char.get(idx, '?'))
        return ''.join(chars)

vocab = Vocab()

def generate_data(n_samples=5000):
    """Generate random strings and their reverses"""
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    data = []
    for _ in range(n_samples):
        length = random.randint(3, MAX_LENGTH - 2)
        src = ''.join(random.choices(chars, k=length))
        tgt = src[::-1]
        vocab.add_string(src)
        vocab.add_string(tgt)
        data.append((src, tgt))
    return data

raw_data = generate_data(5000)
random.shuffle(raw_data)
split = int(0.9 * len(raw_data))
train_pairs = raw_data[:split]
val_pairs = raw_data[split:]

def collate_batch(pairs):
    src_batch = torch.tensor([vocab.encode(s) for s, _ in pairs], dtype=torch.long)
    tgt_batch = torch.tensor([vocab.encode(t) for _, t in pairs], dtype=torch.long)
    return src_batch.to(DEVICE), tgt_batch.to(DEVICE)

# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_TOKEN)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
    
    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: [batch, src_len, hid_dim*2]
        
        # Combine bidirectional states
        hidden = hidden.view(self.rnn.num_layers, 2, src.size(0), -1)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # [n_layers, batch, hid_dim*2]
        cell = cell.view(self.rnn.num_layers, 2, src.size(0), -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        
        # Project to decoder hidden size
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        return outputs, (hidden, cell)

# -----------------------------
# Attention
# -----------------------------
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, dec_hid_dim]
        # encoder_outputs: [batch, src_len, enc_hid_dim]
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# -----------------------------
# Decoder
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_TOKEN)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + enc_hid_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch] -> [batch, 1]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]
        
        # Attention
        a = self.attention(hidden[-1], encoder_outputs)  # [batch, src_len]
        a = a.unsqueeze(1)  # [batch, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs)  # [batch, 1, enc_hid_dim]
        
        rnn_input = torch.cat([embedded, weighted], dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # output: [batch, 1, dec_hid_dim]
        # weighted: [batch, 1, enc_hid_dim]
        # embedded: [batch, 1, emb_dim]
        prediction = self.fc_out(torch.cat([output, weighted, embedded], dim=2))
        return prediction.squeeze(1), hidden, cell, a.squeeze(1)

# -----------------------------
# Seq2Seq Wrapper
# -----------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        input = trg[:, 0]  # <SOS>
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        
        return outputs

# -----------------------------
# Initialize Model
# -----------------------------
INPUT_DIM = vocab.n_chars
OUTPUT_DIM = vocab.n_chars
ENC_HID_DIM = HIDDEN_SIZE * 2  # bidirectional
DEC_HID_DIM = HIDDEN_SIZE

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBEDDING_DIM, ENC_HID_DIM, DEC_HID_DIM, NUM_LAYERS, DROPOUT, attn)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters: {count_parameters(model):,}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

# -----------------------------
# Training & Evaluation
# -----------------------------
def train_epoch(model, data, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0
    random.shuffle(data)
    
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]
        src, trg = collate_batch(batch)
        
        optimizer.zero_grad()
        output = model(src, trg, TEACHER_FORCING_RATIO)
        
        # output: [batch, trg_len, vocab]
        # trg: [batch, trg_len]
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / (len(data) / BATCH_SIZE)

def evaluate(model, data, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            src, trg = collate_batch(batch)
            
            output = model(src, trg, 0.0)  # No teacher forcing
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            # Accuracy
            pred = output.argmax(1)
            mask = trg != PAD_TOKEN
            correct += (pred[mask] == trg[mask]).sum().item()
            total += mask.sum().item()
    
    return epoch_loss / (len(data) / BATCH_SIZE), correct / total if total > 0 else 0

def translate(model, sentence, max_len=MAX_LENGTH):
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor([vocab.encode(sentence)], dtype=torch.long).to(DEVICE)
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor)
        
        trg_indices = [SOS_TOKEN]
        attentions = []
        
        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices[-1]], dtype=torch.long).to(DEVICE)
            output, hidden, cell, attn_weights = model.decoder(
                trg_tensor, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()
            trg_indices.append(pred_token)
            attentions.append(attn_weights.cpu().numpy())
            
            if pred_token == EOS_TOKEN:
                break
        
        return vocab.decode(trg_indices), attentions

# -----------------------------
# Training Loop
# -----------------------------
print("\n--- Training Started ---")
start_time = time.time()
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_pairs, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_pairs, criterion)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_seq2seq.pt')
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

print(f"\nBest Val Loss: {best_val_loss:.4f}")

# -----------------------------
# Qualitative Evaluation
# -----------------------------
print("\n--- Sample Translations ---")
model.load_state_dict(torch.load('best_seq2seq.pt'))
test_samples = [
    "hello world",
    "pytorch",
    "sequence to sequence",
    "123456789",
    "attention mechanism",
    "encoder decoder",
    "deep learning day 108",
]

for src in test_samples:
    pred, attns = translate(model, src)
    status = "✓" if pred == src[::-1] else "✗"
    print(f"  {status} Src: '{src}' -> Pred: '{pred}' (Target: '{src[::-1]}')")

# -----------------------------
# Attention Visualization (Text-based)
# -----------------------------
print("\n--- Attention Heatmap (Sample) ---")
src = "abcdef"
pred, attns = translate(model, src)
print(f"Source:  {src}")
print(f"Target:  {src[::-1]}")
print(f"Predict: {pred}")
print("\nAttention weights (decoder step x source position):")
src_tokens = list(src) + ['<EOS>']
for step, attn in enumerate(attns[:len(src)+1]):
    if step >= len(pred) + 1:
        break
    bar = ''.join(['█' if w > 0.15 else '░' for w in attn[0][:len(src_tokens)]])
    print(f"  Step {step:2d} ({pred[step] if step < len(pred) else 'EOS'}): {bar} {attn[0][:len(src_tokens)]}")

print("\n--- Experiment Complete ---")