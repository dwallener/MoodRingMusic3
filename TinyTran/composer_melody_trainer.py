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
from torch.nn.utils.rnn import pad_sequence


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

# Example Duration Vocabulary (in beats)
DURATION_VOCAB = [0.25, 0.5, 0.75, 1.0]  # Eighth, quarter, dotted quarter, half notes
DURATION_VOCAB_SIZE = len(DURATION_VOCAB)

# Example Register Vocabulary
REGISTER_VOCAB = ['low', 'mid', 'high']
REGISTER_VOCAB_SIZE = len(REGISTER_VOCAB)

# --------------------
# DATA PREPARATION
# --------------------

def extract_melody_features(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        melody_notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                melody_notes.append(note)

        melody_notes.sort(key=lambda n: n.start)
        intervals, durations, registers = [], [], []
        last_pitch = None

        for note in melody_notes:
            if last_pitch is not None:
                interval = note.pitch - last_pitch
                interval = max(INTERVAL_RANGE[0], min(INTERVAL_RANGE[1], interval))
                intervals.append(interval)
            last_pitch = note.pitch

            durations.append(quantize_duration(note.end - note.start))
            registers.append(classify_register(note.pitch))

        return intervals, durations, registers
    except:
        return [], [], []

def quantize_duration(duration):
    # Example: Quantize durations to nearest in [0.25, 0.5, 0.75, 1.0]
    quantized = min(DURATION_VOCAB, key=lambda x: abs(x - duration))
    return quantized

def classify_register(pitch):
    # Example: Map pitch to low, mid, high register classes
    if pitch < 60:
        return 'low'
    elif pitch < 72:
        return 'mid'
    else:
        return 'high'

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

"""
def build_dataset():
    meta = pd.read_csv(META_CSV)
    files = meta[meta['composer'].str.contains(COMPOSER, na=False)]['midi_filename'].tolist()
    sequences = []

    for fname in tqdm(files):
        intervals, durations, registers = extract_melody_features(os.path.join(MIDI_BASE, fname))

        if len(intervals) >= SEED_LENGTH + 32:
            for i in range(len(intervals) - (SEED_LENGTH + 32)):
                seed = intervals[i:i+SEED_LENGTH]
                target_intervals = intervals[i+SEED_LENGTH:i+SEED_LENGTH+32]
                target_durations = durations[i+SEED_LENGTH:i+SEED_LENGTH+32]
                target_registers = registers[i+SEED_LENGTH:i+SEED_LENGTH+32]
                sequences.append((seed, target_intervals, target_durations, target_registers))

    return sequences
"""

def build_dataset(target_beats=128, tolerance=5.0):
    meta = pd.read_csv(META_CSV)
    files = meta[meta['composer'].str.contains(COMPOSER, na=False)]['midi_filename'].tolist()
    sequences = []

    for fname in tqdm(files):
        intervals, durations, registers = extract_melody_features(os.path.join(MIDI_BASE, fname))

        # Skip if not enough data in this file
        if len(durations) < SEED_LENGTH:
            continue

        for i in range(len(durations) - SEED_LENGTH):
            cum_dur = 0.0
            j = i + SEED_LENGTH
            while j < len(durations) and cum_dur < target_beats:
                cum_dur += durations[j]
                j += 1

            # Check if cumulative duration is within tolerance
            if abs(cum_dur - target_beats) <= tolerance:
                seed = intervals[i:i+SEED_LENGTH]
                target_intervals = intervals[i+SEED_LENGTH:j]
                target_durations = durations[i+SEED_LENGTH:j]
                target_registers = registers[i+SEED_LENGTH:j]
                sequences.append((seed, target_intervals, target_durations, target_registers))

    return sequences

# --------------------
# MODEL DEFINITION
# --------------------

class TinyMelodyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.interval_head = nn.Linear(128, VOCAB_SIZE)             # Predict Intervals
        self.duration_head = nn.Linear(128, DURATION_VOCAB_SIZE)    # Predict Durations
        self.register_head = nn.Linear(128, REGISTER_VOCAB_SIZE)    # Predict Register Classes

    """
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.interval_head(x), self.duration_head(x), self.register_head(x)
    """

    def forward(self, x, attention_mask=None):
        x = self.embed(x)

        if attention_mask is not None:
            # Transformer expects mask shape: (batch_size, sequence_length)
            # Convert to (sequence_length, sequence_length) causal mask if necessary
            # But since this is a padding mask, use directly:
            x = self.transformer(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.transformer(x)

        return self.interval_head(x), self.duration_head(x), self.register_head(x)

# --------------------
# TRAINING LOOP
# --------------------

def train():
    sequences = build_dataset()  # Should now include (seed, target_intervals, target_durations, target_registers)
    random.shuffle(sequences)
    split = int(0.8 * len(sequences))
    train_data = sequences[:split]
    val_data = sequences[split:]

    model = TinyMelodyTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn_interval = nn.CrossEntropyLoss(ignore_index=-1)
    loss_fn_duration = nn.CrossEntropyLoss(ignore_index=-1)
    loss_fn_register = nn.CrossEntropyLoss(ignore_index=-1)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 1
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 1)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0
        batch_inputs, batch_intervals, batch_durations, batch_registers = [], [], [], []

        for idx, (seed, target_intervals, target_durations, target_registers) in enumerate(tqdm(train_data)):
            input_seq = seed + target_intervals[:-1]
            #input_seq = seed

            input_tensor = torch.tensor(
                [INTERVAL_VOCAB.index(i) for i in input_seq], dtype=torch.long
            ) # .unsqueeze(0)
            interval_tensor = torch.tensor(
                [INTERVAL_VOCAB.index(i) for i in target_intervals], dtype=torch.long
            )# .unsqueeze(0)
            duration_tensor = torch.tensor(
                [DURATION_VOCAB.index(i) for i in target_durations], dtype=torch.long
            )# .unsqueeze(0)
            register_tensor = torch.tensor(
                [REGISTER_VOCAB.index(i) for i in target_registers], dtype=torch.long
            )# .unsqueeze(0)

            batch_inputs.append(input_tensor)
            batch_intervals.append(interval_tensor)
            batch_durations.append(duration_tensor)
            batch_registers.append(register_tensor)

            if len(batch_inputs) == BATCH_SIZE or idx == len(train_data) - 1:
                batch_inputs_tensor = pad_sequence(batch_inputs, batch_first=True, padding_value=INTERVAL_PAD).to(DEVICE)
                batch_intervals_tensor = pad_sequence(batch_intervals, batch_first=True, padding_value=-1).to(DEVICE)
                batch_durations_tensor = pad_sequence(batch_durations, batch_first=True, padding_value=-1).to(DEVICE)
                batch_registers_tensor = pad_sequence(batch_registers, batch_first=True, padding_value=-1).to(DEVICE)

                # Create Attention Mask (True for valid tokens, False for padding)
                attention_mask = (batch_inputs_tensor != INTERVAL_PAD).to(torch.bool)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():

                    # Check Shapes
                    print(f"batch_inputs_tensor.shape: {batch_inputs_tensor.shape}")
                    print(f"batch_durations_tensor.shape: {batch_durations_tensor.shape}")

                    # Model Call    
                    pred_intervals, pred_durations, pred_registers = model(
                        batch_inputs_tensor, attention_mask=attention_mask
                    )

                    print(f"pred_durations.shape: {pred_durations.shape}")

                    # Align predictions with target sequence lengths before applying masks
                    pred_intervals = pred_intervals[:, -batch_intervals_tensor.size(1):, :]
                    pred_durations = pred_durations[:, -batch_durations_tensor.size(1):, :]
                    pred_registers = pred_registers[:, -batch_registers_tensor.size(1):, :]

                    # Create valid masks for each target
                    valid_intervals = batch_intervals_tensor != -1
                    valid_durations = batch_durations_tensor != -1
                    valid_registers = batch_registers_tensor != -1

                    # Flatten predictions and targets based on valid masks
                    pred_intervals_flat = pred_intervals[valid_intervals]
                    batch_intervals_flat = batch_intervals_tensor[valid_intervals]

                    pred_durations_flat = pred_durations[valid_durations]
                    batch_durations_flat = batch_durations_tensor[valid_durations]

                    pred_registers_flat = pred_registers[valid_registers]
                    batch_registers_flat = batch_registers_tensor[valid_registers]

                    # Compute losses
                    loss_interval = loss_fn_interval(pred_intervals_flat, batch_intervals_flat)
                    loss_duration = loss_fn_duration(pred_durations_flat, batch_durations_flat)
                    loss_register = loss_fn_register(pred_registers_flat, batch_registers_flat)

                    total_batch_loss = loss_interval + loss_duration + loss_register

                # Assume center pitch is Middle C (MIDI 60)
                # We're trying to rationalize the note register here - can't be too high or too lowSo now 
                center_pitch = 60
                # Reconstruct predicted pitches from intervals
                predicted_pitches = center_pitch + torch.cumsum(batch_intervals_tensor.float(), dim=1)
                # Calculate mean squared deviation from center
                register_penalty_weight = 0.005  # Try values from 0.001 to 0.01
                register_penalty = torch.mean((predicted_pitches - center_pitch) ** 2) * register_penalty_weight
                #register_penalty = torch.mean((predicted_pitches - center_pitch) ** 2) * 0.001  # Adjust weight as needed

                total_batch_loss = loss_interval + loss_duration + loss_register + register_penalty                

                scaler.scale(total_batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += total_batch_loss.item()
                batch_inputs, batch_intervals, batch_durations, batch_registers = [], [], [], []

        print(f"Epoch {epoch} Loss: {total_loss/len(train_data):.4f}")

        if epoch % CHECKPOINT_INTERVAL == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            checkpoint_path = os.path.join(SAVE_DIR, f"{COMPOSER}_epoch{epoch}.pt")
            #torch.save(model.state_dict(), checkpoint_path)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)

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
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument(
        "--register_penalty_weight", 
        type=float, 
        default=0.001, 
        help="Weight for register drift penalty in loss function (default: 0.001)"
)
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
