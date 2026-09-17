import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Tuple

torch.manual_seed(42)
random.seed(42)

VOCAB_SIZE = 1000
HIDDEN_SIZE = 128
NUM_LAYERS = 3
NUM_HEADS = 4
INTERMEDIATE_SIZE = 512
MAX_SEQ_LEN = 64
DROPOUT = 0.1
MASK_TOKEN_ID = 3
CLS_TOKEN_ID = 1
SEP_TOKEN_ID = 2
PAD_TOKEN_ID = 0

class BertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, padding_idx=PAD_TOKEN_ID)
        self.position_embeddings = nn.Embedding(MAX_SEQ_LEN, HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(2, HIDDEN_SIZE)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embeddings = (self.word_embeddings(input_ids) + 
                      self.position_embeddings(position_ids) + 
                      self.token_type_embeddings(token_type_ids))
        return self.dropout(self.LayerNorm(embeddings))

class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_size = HIDDEN_SIZE // NUM_HEADS
        self.all_head_size = self.num_heads * self.head_size
        self.query = nn.Linear(HIDDEN_SIZE, self.all_head_size)
        self.key = nn.Linear(HIDDEN_SIZE, self.all_head_size)
        self.value = nn.Linear(HIDDEN_SIZE, self.all_head_size)
        self.dropout = nn.Dropout(DROPOUT)

    def transpose_for_scores(self, x: torch.Tensor):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_shape)

class BertSelfOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)

class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        self_output = self.self(hidden_states, attention_mask)
        return self.output(self_output, hidden_states)

class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)

    def forward(self, hidden_states: torch.Tensor):
        return F.gelu(self.dense(hidden_states))

class BertOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)

class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)

class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer() for _ in range(NUM_LAYERS)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.dense(hidden_states[:, 0]))

class BertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder = BertEncoder()
        self.pooler = BertPooler()

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None, attention_mask: torch.Tensor = None):
        if attention_mask is None:
            attention_mask = (input_ids != PAD_TOKEN_ID).float()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

class BertForMaskedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel()
        self.cls = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None, attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        prediction_scores = self.cls(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prediction_scores.view(-1, VOCAB_SIZE), labels.view(-1))
        return loss, prediction_scores

def create_masked_lm_data(batch_size: int, seq_len: int, mask_prob: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(4, VOCAB_SIZE, (batch_size, seq_len))
    input_ids[:, 0] = CLS_TOKEN_ID
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mask_prob)
    special_tokens_mask = (labels <= 3).float()
    probability_matrix.masked_fill_(special_tokens_mask.bool(), 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = MASK_TOKEN_ID
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(4, VOCAB_SIZE, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    return input_ids, labels

def train_step(model: BertForMaskedLM, optimizer: torch.optim.Optimizer, input_ids: torch.Tensor, labels: torch.Tensor):
    model.train()
    optimizer.zero_grad()
    loss, _ = model(input_ids, labels=labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

def evaluate(model: BertForMaskedLM, input_ids: torch.Tensor, labels: torch.Tensor):
    model.eval()
    with torch.no_grad():
        loss, logits = model(input_ids, labels=labels)
        predictions = logits.argmax(dim=-1)
        mask = labels != -100
        correct = (predictions[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0
    return loss.item(), acc

def demo_inference(model: BertForMaskedLM):
    model.eval()
    text = "the quick brown fox jumps over the lazy dog"
    tokens = text.split()
    token_ids = [CLS_TOKEN_ID] + [hash(t) % (VOCAB_SIZE - 4) + 4 for t in tokens] + [SEP_TOKEN_ID]
    mask_pos = 4
    original = token_ids[mask_pos]
    token_ids[mask_pos] = MASK_TOKEN_ID
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        _, logits = model(input_ids)
        pred_id = logits[0, mask_pos].argmax().item()
    id_to_token = {v: k for k, v in enumerate(tokens, 4)}
    id_to_token.update({CLS_TOKEN_ID: "[CLS]", SEP_TOKEN_ID: "[SEP]", MASK_TOKEN_ID: "[MASK]", PAD_TOKEN_ID: "[PAD]"})
    print(f"Original: {text}")
    print(f"Masked:   {' '.join([id_to_token.get(t, f'[UNK{t}]') for t in token_ids])}")
    print(f"Predicted: {id_to_token.get(pred_id, f'[UNK{pred_id}]')} (true: {id_to_token.get(original, f'[UNK{original}]')})")

def main():
    print("=" * 60)
    print("Day 71: BERT Masked Language Modeling Mini-Experiment")
    print("=" * 60)
    model = BertForMaskedLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print("\nTraining on synthetic data...")
    for epoch in range(5):
        epoch_loss = 0
        for step in range(20):
            input_ids, labels = create_masked_lm_data(16, 32)
            loss = train_step(model, optimizer, input_ids, labels)
            epoch_loss += loss
        avg_loss = epoch_loss / 20
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")
    print("\nEvaluating...")
    eval_loss, eval_acc = evaluate(model, *create_masked_lm_data(32, 32))
    print(f"Eval loss: {eval_loss:.4f}, Masked token accuracy: {eval_acc:.4f}")
    print("\nInference demo:")
    demo_inference(model)
    print("\nExperiment complete.")

if __name__ == "__main__":
    main()