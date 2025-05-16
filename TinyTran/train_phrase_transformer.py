import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# ----- Dataset Loader -----
class PhraseDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.phrases = json.load(f)

        tokens = {token for phrase in self.phrases for token in phrase}
        self.vocab = {token: idx + 1 for idx, token in enumerate(sorted(tokens))}
        self.vocab["<PAD>"] = 0
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        token_ids = [self.vocab[token] for token in phrase]
        return torch.tensor(token_ids, dtype=torch.long)

    def get_vocab_size(self):
        return len(self.vocab)

def collate_fn(batch):
    return nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

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

# ----- Training Function -----
def train_model(json_file, num_epochs=20, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PhraseDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    vocab_size = dataset.get_vocab_size()

    model = TinyTransformer(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

    # Save model state
    torch.save(model.state_dict(), "tiny_transformer_phrase_dataset_long.json.pt")

    # Save vocab mappings
    with open("tiny_transformer_phrase_dataset_long.vocab.json", "w") as f:
        json.dump({
            "token_to_idx": dataset.vocab, 
            "idx_to_token": dataset.inv_vocab}, f)

    print("Model and vocab saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_phrase_transformer.py <phrase_dataset.json>")
    else:
        train_model(sys.argv[1])