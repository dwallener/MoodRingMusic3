import torch
import torch.nn as nn
import json
import sys

# ----- Tiny Transformer -----
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embed(x)  # (B, T, D)
        emb = emb.permute(1, 0, 2)  # (T, B, D) for Transformer
        out = self.transformer(emb)
        out = out.permute(1, 0, 2)  # (B, T, D)
        logits = self.fc_out(out)   # (B, T, Vocab_Size)
        return logits

# ----- Load Model -----
def load_model(model_path, vocab_size):
    model = TinyTransformer(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ----- Load Vocab -----
def load_vocab(vocab_file):
    with open(vocab_file, "r") as f:
        vocab_data = json.load(f)
    return vocab_data["token_to_idx"], vocab_data["idx_to_token"]

# ----- Phrase Generation -----
def generate_phrase(model, token_to_idx, idx_to_token, max_length=16, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_token = list(token_to_idx.values())[1]  # Use the first valid token as a start
    sequence = [start_token]

    for _ in range(max_length - 1):
        seq_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(seq_tensor)  # Shape: (1, T, vocab_size)
            last_logits = logits[0, -1, :]  # Get logits for last position
            probabilities = torch.softmax(last_logits, dim=-1)

            # Apply temperature scaling
            probabilities = probabilities ** (1.0 / temperature)
            probabilities = probabilities / probabilities.sum()

            # Sample next token
            next_token = torch.multinomial(probabilities, 1).item()

        if next_token == 0:  # PAD token, stop generation
            break
        sequence.append(next_token)

    # Convert token IDs back to actual tokens
    return [idx_to_token[str(idx)] for idx in sequence]

# ----- Main -----
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_phrase.py <model_file> <vocab_file> <phrase_length> [temperature]")
        sys.exit(1)

    model_file = sys.argv[1]
    vocab_file = sys.argv[2]
    max_length = int(sys.argv[3])
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    token_to_idx, idx_to_token = load_vocab(vocab_file)
    model = load_model(model_file, len(token_to_idx))

    generated_tokens = generate_phrase(model, token_to_idx, idx_to_token, max_length=max_length, temperature=temperature)

    print("Generated Phrase Tokens:", generated_tokens)