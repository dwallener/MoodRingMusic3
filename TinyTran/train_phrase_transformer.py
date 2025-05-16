import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle

# Constants
VOCAB_SIZE = 25 * len([0.25, 0.5, 1.0, 2.0, 4.0])  # (Interval -12 to +12) * 5 duration buckets = 125
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Model Definition
# ---------------------------
class PhraseTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(PhraseTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=NUM_HEADS)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)  # Shape: [seq_len, batch, embed_dim]
        out = self.transformer(emb)
        logits = self.fc_out(out)
        return logits

# ---------------------------
# Dataset Preparation
# ---------------------------
def load_dataset(file_path):
    with open(file_path, "rb") as f:
        phrases = pickle.load(f)
    return phrases

def generate_batches(phrases, batch_size):
    while True:
        batch = random.sample(phrases, batch_size)
        inputs = []
        targets = []
        for seq in batch:
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            target_seq = seq[1:]
            inputs.append(torch.tensor(input_seq, dtype=torch.long))
            targets.append(torch.tensor(target_seq, dtype=torch.long))
        
        if not inputs:
            continue  # Skip empty batch

        inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=False)
        targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=False)
        yield inputs_padded.to(DEVICE), targets_padded.to(DEVICE)

# ---------------------------
# Training Loop
# ---------------------------
def train_model(phrases, output_model_file):
    model = PhraseTransformer(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_generator = generate_batches(phrases, BATCH_SIZE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        inputs, targets = next(batch_generator)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Reshape for loss: (seq_len * batch_size, vocab_size)
        loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{EPOCHS} - Loss: {loss.item():.4f}")

    # Save final model
    torch.save(model.state_dict(), output_model_file)
    print(f"Model saved to {output_model_file}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python train_phrase_transformer.py <dataset_pickle> <output_model_file>")
    else:
        phrases = load_dataset(sys.argv[1])
        train_model(phrases, sys.argv[2])


