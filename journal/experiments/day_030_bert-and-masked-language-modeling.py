import torch
from transformers import BertTokenizer, BertForMaskedLM

def main():
    # Use a tiny BERT model for quick demonstration
    model_name = "prajjwal1/bert-tiny"  # 2 layers, 128 hidden, 2 heads, ~4.4M params
    print(f"Loading {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()

    # Sample sentence
    text = "The quick brown fox jumps over the lazy dog."
    print(f"\nOriginal text: {text}")

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Token IDs: {input_ids.tolist()[0]}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

    # Create masked version: mask a random token (not special tokens)
    special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids[0], already_has_special_tokens=True)
    candidate_indices = [i for i, mask in enumerate(special_tokens_mask) if mask == 0]
    if not candidate_indices:
        print("No non-special tokens to mask.")
        return
    import random
    mask_idx = random.choice(candidate_indices)
    masked_input_ids = input_ids.clone()
    masked_input_ids[0, mask_idx] = tokenizer.mask_token_id
    print(f"\nMasked token index: {mask_idx}")
    print(f"Masked tokens: {tokenizer.convert_ids_to_tokens(masked_input_ids[0])}")

    # Predict
    with torch.no_grad():
        outputs = model(masked_input_ids)
        logits = outputs.logits
    mask_logits = logits[0, mask_idx]
    probs = torch.softmax(mask_logits, dim=-1)
    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)
    print(f"\nTop {top_k} predictions for masked token:")
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
        print(f"  {token}: {prob.item():.4f}")

    # Show original token
    original_token = tokenizer.convert_ids_to_tokens([input_ids[0, mask_idx].item()])[0]
    print(f"\nOriginal token was: {original_token}")

if __name__ == "__main__":
    main()