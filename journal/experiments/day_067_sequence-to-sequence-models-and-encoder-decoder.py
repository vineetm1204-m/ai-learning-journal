import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ============================================================
# Day 67: Sequence-to-Sequence Encoder-Decoder Mini-Experiment
# Task: Character-level sequence reversal (input: "abc" -> output: "cba")
# ============================================================

# ---- Config ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = list("abcdefghijklmnopqrstuvwxyz ")
VOCAB_SIZE = len(VOCAB) + 2  # +SOS, +EOS
SOS_IDX = VOCAB_SIZE - 2
EOS_IDX = VOCAB_SIZE - 1
PAD_IDX = 0  # not used but reserved
MAX_LEN = 12
BATCH_SIZE = 64
HIDDEN_SIZE = 128
EMBED_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 30
TEACHER_FORCING_RATIO = 0.5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---- Vocab maps ----
char2idx = {c: i for i, c in enumerate(VOCAB)}
char2idx['<SOS>'] = SOS_IDX
char2idx['<EOS>'] = EOS_IDX
idx2char = {i: c for c, i in char2idx.items()}

# ---- Synthetic Data: Sequence Reversal ----
def generate_reversal_data(num_samples=5000, max_len=MAX_LEN):
    src, tgt = [], []
    for _ in range(num_samples):
        length = random.randint(1, max_len)
        seq = [random.choice(VOCAB[:-1]) for _ in range(length)]  # no space for simplicity
        src.append(seq)
        tgt.append(list(reversed(seq)))
    return src, tgt

def encode(seq, add_sos=False, add_eos=True):
    idxs = [char2idx[c] for c in seq]
    if add_sos:
        idxs = [SOS_IDX] + idxs
    if add_eos:
        idxs = idxs + [EOS_IDX]
    return idxs

def collate_batch(src_seqs, tgt_seqs):
    src_tensor = [torch.tensor(encode(s), dtype=torch.long) for s in src_seqs]
    tgt_tensor = [torch.tensor(encode(t, add_sos=True, add_eos=True), dtype=torch.long) for t in tgt_seqs]
    src_padded = nn.utils.rnn.pad_sequence(src_tensor, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_tensor, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded

train_src, train_tgt = generate_reversal_data(4000)
val_src, val_tgt = generate_reversal_data(500)
test_src, test_tgt = generate_reversal_data(500)

# ---- Model: Encoder ----
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, src, src_lengths):
        # src: [B, T_src]
        embedded = self.dropout(self.embedding(src))  # [B, T, E]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # [B, T, 2*H]
        # Combine bidirectional: sum forward/backward
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # hidden: [2*num_layers, B, H] -> combine directions
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size).sum(dim=1)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size).sum(dim=1)
        return outputs, hidden, cell

# ---- Attention ----
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: [B, H], encoder_outputs: [B, T, H]
        # score = Va * tanh(Wa * h_enc + Ua * h_dec)
        Wa_out = self.Wa(encoder_outputs)  # [B, T, H]
        Ua_out = self.Ua(decoder_hidden).unsqueeze(1)  # [B, 1, H]
        scores = self.Va(torch.tanh(Wa_out + Ua_out)).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, H]
        return context, attn_weights

# ---- Decoder ----
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, attention):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.attention = attention
        self.rnn = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input_token, hidden, cell, encoder_outputs, src_mask):
        # input_token: [B] (single token)
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # [B, 1, E]
        # Attention
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, src_mask)  # [B, H], [B, T]
        # Concat embedded + context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)  # [B, 1, E+H]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [B, 1, H]
        # Concat output + context for prediction
        output = output.squeeze(1)  # [B, H]
        pred_input = torch.cat([output, context], dim=-1)  # [B, 2H]
        prediction = self.fc_out(pred_input)  # [B, V]
        return prediction, hidden, cell, attn_weights

# ---- Seq2Seq Wrapper ----
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_src_mask(self, src, src_lengths):
        # src: [B, T], mask: [B, T] (1 for valid, 0 for pad)
        max_len = src.size(1)
        mask = torch.arange(max_len, device=self.device).expand(src.size(0), max_len) < src_lengths.unsqueeze(1)
        return mask.float()

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        # src: [B, T_src], tgt: [B, T_tgt] (with SOS)
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        src_mask = self.create_src_mask(src, src_lengths)

        input_token = tgt[:, 0]  # SOS
        for t in range(1, tgt_len):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, src_mask)
            outputs[:, t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1
        return outputs

    def translate(self, src, src_lengths, max_len=MAX_LEN):
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            src_mask = self.create_src_mask(src, src_lengths)
            batch_size = src.size(0)
            input_token = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)
            decoded = []
            for _ in range(max_len):
                prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, src_mask)
                top1 = prediction.argmax(1)
                decoded.append(top1)
                input_token = top1
                if (top1 == EOS_IDX).all():
                    break
            return torch.stack(decoded, dim=1)  # [B, T_out]

# ---- Instantiate ----
encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
attention = BahdanauAttention(HIDDEN_SIZE).to(DEVICE)
decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, attention).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# ---- Training / Evaluation ----
def train_epoch(model, src_data, tgt_data, optimizer, criterion, teacher_forcing_ratio):
    model.train()
    total_loss = 0
    indices = list(range(len(src_data)))
    random.shuffle(indices)
    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        src_batch = [train_src[j] for j in batch_idx]
        tgt_batch = [train_tgt[j] for j in batch_idx]
        src_tensor, tgt_tensor = collate_batch(src_batch, tgt_batch)
        src_tensor, tgt_tensor = src_tensor.to(DEVICE), tgt_tensor.to(DEVICE)
        src_lengths = (src_tensor != PAD_IDX).sum(dim=1)

        optimizer.zero_grad()
        output = model(src_tensor, src_lengths, tgt_tensor, teacher_forcing_ratio)
        # output: [B, T, V], tgt: [B, T] -> shift
        loss = criterion(output[:, 1:].reshape(-1, VOCAB_SIZE), tgt_tensor[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (len(indices) / BATCH_SIZE)

def evaluate(model, src_data, tgt_data, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(src_data), BATCH_SIZE):
            src_batch = src_data[i:i+BATCH_SIZE]
            tgt_batch = tgt_data[i:i+BATCH_SIZE]
            src_tensor, tgt_tensor = collate_batch(src_batch, tgt_batch)
            src_tensor, tgt_tensor = src_tensor.to(DEVICE), tgt_tensor.to(DEVICE)
            src_lengths = (src_tensor != PAD_IDX).sum(dim=1)
            output = model(src_tensor, src_lengths, tgt_tensor, teacher_forcing_ratio=0.0)
            loss = criterion(output[:, 1:].reshape(-1, VOCAB_SIZE), tgt_tensor[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / (len(src_data) / BATCH_SIZE)

def compute_accuracy(model, src_data, tgt_data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(src_data), BATCH_SIZE):
            src_batch = src_data[i:i+BATCH_SIZE]
            tgt_batch = tgt_data[i:i+BATCH_SIZE]
            src_tensor, _ = collate_batch(src_batch, tgt_batch)
            src_tensor = src_tensor.to(DEVICE)
            src_lengths = (src_tensor != PAD_IDX).sum(dim=1)
            preds = model.translate(src_tensor, src_lengths)
            for pred, target in zip(preds, tgt_batch):
                pred_str = ''.join(idx2char.get(p.item(), '') for p in pred if p.item() not in (SOS_IDX, EOS_IDX, PAD_IDX))
                tgt_str = ''.join(target)
                if pred_str == tgt_str:
                    correct += 1
                total += 1
    return correct / total

# ---- Training Loop ----
print(f"Device: {DEVICE}")
print(f"Vocab size: {VOCAB_SIZE}, Hidden: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}")
print("Starting training...\n")

train_losses, val_losses, val_accs = [], [], []
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_src, train_tgt, optimizer, criterion, TEACHER_FORCING_RATIO)
    val_loss = evaluate(model, val_src, val_tgt, criterion)
    val_acc = compute_accuracy(model, val_src, val_tgt)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ---- Test Evaluation ----
test_acc = compute_accuracy(model, test_src, test_tgt)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ---- Qualitative Examples ----
print("\n--- Sample Predictions ---")
model.eval()
with torch.no_grad():
    for i in range(10):
        src_seq = test_src[i]
        tgt_seq = test_tgt[i]
        src_tensor, _ = collate_batch([src_seq], [tgt_seq])
        src_tensor = src_tensor.to(DEVICE)
        src_lengths = (src_tensor != PAD_IDX).sum(dim=1)
        pred = model.translate(src_tensor, src_lengths)[0].cpu().tolist()
        pred_str = ''.join(idx2char.get(p, '') for p in pred if p not in (SOS_IDX, EOS_IDX, PAD_IDX))
        src_str = ''.join(src_seq)
        tgt_str = ''.join(tgt_seq)
        status = "✓" if pred_str == tgt_str else "✗"
        print(f"  {status} Src: '{src_str}' | Tgt: '{tgt_str}' | Pred: '{pred_str}'")

# ---- Plot Learning Curves ----
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Seq2Seq Learning Curves (Reversal Task)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Sequence-Level Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('day67_seq2seq_curves.png', dpi=150)
print("\nPlot saved to day67_seq2seq_curves.png")

# ---- Attention Visualization (single example) ----
print("\n--- Attention Weights (Example) ---")
model.eval()
with torch.no_grad():
    src_seq = list("hello")
    tgt_seq = list("olleh")
    src_tensor, _ = collate_batch([src_seq], [tgt_seq])
    src_tensor = src_tensor.to(DEVICE)
    src_lengths = (src_tensor != PAD_IDX).sum(dim=1)
    encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lengths)
    src_mask = model.create_src_mask(src_tensor, src_lengths)
    input_token = torch.tensor([SOS_IDX], device=DEVICE)
    attentions = []
    for _ in range(len(tgt_seq) + 1):
        prediction, hidden, cell, attn = model.decoder(input_token, hidden, cell, encoder_outputs, src_mask)
        attentions.append(attn[0].cpu().numpy())
        top1 = prediction.argmax(1)
        input_token = top1
        if top1.item() == EOS_IDX:
            break
    attentions = np.array(attentions)  # [T_out, T_src]
    print(f"Input:  {' '.join(src_seq)}")
    print(f"Output: {' '.join(tgt_seq)}")
    print("Attention matrix (rows=decoder steps, cols=encoder steps):")
    print(np.round(attentions, 3))