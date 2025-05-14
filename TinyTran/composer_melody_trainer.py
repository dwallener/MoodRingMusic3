# composer_melody_trainer.py

import pandas as pd
import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pretty_midi

# --------------------
# CONFIGURATION (Defaults)
# --------------------
META_CSV = "annotated_csv.csv"
MIDI_BASE = "maestro-v3.0.0/"
COMPOSER = "Chopin"
SAVE_DIR = "checkpoints"
EPOCHS = 500
CHECKPOINT_INTERVAL = 100
SEED_LENGTH = 4
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Tokenization Parameters
INTERVAL_RANGE = (-24, 24)
INTERVAL_VOCAB = [i for i in range(INTERVAL_RANGE[0], INTERVAL_RANGE[1] + 1)]
INTERVAL_PAD = len(INTERVAL_VOCAB)
VOCAB_SIZE = len(INTERVAL_VOCAB) + 1

# --------------------
# DATA PREPARATION
# --------------------

def extract_melody_intervals(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        melody_notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                melody_notes.append((note.start, note.pitch))
        
        melody_notes.sort()
        intervals = []
        last_pitch = None
        for _, pitch in melody_notes:
            if last_pitch is not None:
                interval = pitch - last_pitch
                interval = max(INTERVAL_RANGE[0], min(INTERVAL_RANGE[1], interval))
                intervals.append(interval)
            last_pitch = pitch
        return intervals
    except:
        return []

def build_dataset():
    meta = pd.read_csv(META_CSV)
    files = meta[meta['composer'].str.contains(COMPOSER, na=False)]['midi_filename'].tolist()
    sequences = []
    for fname in tqdm(files):
        intervals = extract_melody_intervals(os.path.join(MIDI_BASE, fname))
        if len(intervals) >= SEED_LENGTH + 32:
            for i in range(len(intervals) - (SEED_LENGTH + 32)):
                seed = intervals[i:i+SEED_LENGTH]
                target = intervals[i+SEED_LENGTH:i+SEED_LENGTH+32]
                sequences.append((seed, target))
    return sequences

# --------------------
# MODEL DEFINITION
# --------------------

class TinyMelodyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(128, VOCAB_SIZE)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.fc_out(x)

# --------------------
# TRAINING LOOP
# --------------------

def train():
    sequences = build_dataset()
    random.shuffle(sequences)
    split = int(0.8 * len(sequences))
    train_data = sequences[:split]
    val_data = sequences[split:]

    model = TinyMelodyTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # set up for AMP (mixed precision)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        batch_inputs, batch_targets = [], []

        for idx, (seed, target) in enumerate(tqdm(train_data)):
            input_seq = seed + target[:-1]
            target_seq = target

            input_tensor = torch.tensor(
                [INTERVAL_VOCAB.index(i) for i in input_seq], dtype=torch.long
            ).unsqueeze(0)  # Add batch dim
            target_tensor = torch.tensor(
                [INTERVAL_VOCAB.index(i) for i in target_seq], dtype=torch.long
            ).unsqueeze(0)  # Add batch dim

            batch_inputs.append(input_tensor)
            batch_targets.append(target_tensor)

            if len(batch_inputs) == BATCH_SIZE or idx == len(train_data) - 1:
                batch_inputs_tensor = torch.cat(batch_inputs).to(DEVICE)  # Shape: (BATCH_SIZE, seq_len)
                batch_targets_tensor = torch.cat(batch_targets).to(DEVICE)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(batch_inputs_tensor)
                    output = output[:, -batch_targets_tensor.size(1):, :]  # Align last N predictions
                    loss = loss_fn(output.reshape(-1, VOCAB_SIZE), batch_targets_tensor.reshape(-1))
                #loss.backward()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #optimizer.step()

                total_loss += loss.item()
                batch_inputs, batch_targets = [], []

        print(f"Epoch {epoch} Loss: {total_loss/len(train_data):.4f}")

        if epoch % CHECKPOINT_INTERVAL == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            checkpoint_path = os.path.join(SAVE_DIR, f"{COMPOSER}_epoch{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)

        validate(model, val_data)

# --------------------
# VALIDATION LOOP
# --------------------

def validate(model, val_data):
    model.eval()
    with torch.no_grad():
        seed, _ = random.choice(val_data)
        print("Seed Phrase:", seed)

        generated = seed[:]
        input_seq = torch.tensor([INTERVAL_VOCAB.index(i) for i in seed], dtype=torch.long).unsqueeze(1).to(DEVICE)

        for _ in range(32):
            output = model(input_seq).squeeze(1)
            next_token_logits = output[-1]
            next_token = torch.argmax(next_token_logits).item()
            next_interval = INTERVAL_VOCAB[next_token] if next_token < len(INTERVAL_VOCAB) else 0

            generated.append(next_interval)
            input_seq = torch.cat([input_seq, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=0)

        print("Generated Continuation:", generated)

# --------------------
# CHECKPOINT EVALUATION
# --------------------

def evaluate_checkpoint(checkpoint_path):
    print(f"\nLoading checkpoint from {checkpoint_path}")
    model = TinyMelodyTransformer().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    sequences = build_dataset()
    split = int(0.8 * len(sequences))
    val_data = sequences[split:]
    validate(model, val_data)

if __name__ == "__main__":

    # check for CUDA and log it
    # import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device Found")

    parser = argparse.ArgumentParser(description="Tiny Composer Melody Transformer Trainer")
    parser.add_argument("--meta_csv", type=str, default=META_CSV, help="Path to metadata CSV")
    parser.add_argument("--midi_base", type=str, default=MIDI_BASE, help="Base path to MIDI files")
    parser.add_argument("--composer", type=str, default=COMPOSER, help="Composer name to filter by")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Checkpoint save directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL, help="Checkpoint save interval")
    parser.add_argument("--seed_length", type=int, default=SEED_LENGTH, help="Number of tokens in the seed phrase")
    parser.add_argument("--evaluate", type=str, default=None, help="Path to checkpoint file to evaluate only")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")

    args = parser.parse_args()

    META_CSV = args.meta_csv
    MIDI_BASE = args.midi_base
    COMPOSER = args.composer
    SAVE_DIR = args.save_dir
    EPOCHS = args.epochs
    CHECKPOINT_INTERVAL = args.checkpoint_interval
    SEED_LENGTH = args.seed_length
    BATCH_SIZE = args.batch_size

    if args.evaluate:
        evaluate_checkpoint(args.evaluate)
    else:
        train()
