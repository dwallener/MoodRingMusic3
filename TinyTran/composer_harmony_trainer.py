import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pretty_midi

# Configurations
META_CSV = "annotated_csv.csv"
MIDI_BASE = "maestro-v3.0.0/"
SAVE_DIR = "checkpoints"
EPOCHS = 500
CHECKPOINT_INTERVAL = 100
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocab Definitions
CHORD_TYPES = [
    'major', 'minor', 'diminished', 'augmented', 'sus2', 'sus4',
    'major7', 'minor7', 'dominant7', 'half-diminished7', 'diminished7'
]
CHORD_VOCAB_SIZE = len(CHORD_TYPES)
KEY_VOCAB_SIZE = 24
MODE_VOCAB = ['major', 'minor']
MODE_VOCAB_SIZE = len(MODE_VOCAB)

# Helper Function for Chord Type Detection
def detect_chord_type(notes):
    if len(notes) < 3:
        return CHORD_TYPES.index('major')  # Default to 'major' if too few notes

    pitch_classes = sorted(set([note.pitch % 12 for note in notes]))

    intervals = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'diminished': [0, 3, 6],
        'augmented': [0, 4, 8],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'major7': [0, 4, 7, 11],
        'minor7': [0, 3, 7, 10],
        'dominant7': [0, 4, 7, 10],
        'half-diminished7': [0, 3, 6, 10],
        'diminished7': [0, 3, 6, 9],
    }

    for chord_name, interval_pattern in intervals.items():
        root_pitch = pitch_classes[0]
        expected_pitches = [(root_pitch + i) % 12 for i in interval_pattern]
        if all(pc in pitch_classes for pc in expected_pitches):
            return CHORD_TYPES.index(chord_name)

    return CHORD_TYPES.index('major')  # Fallback to 'major'

def detect_style(notes):
    # Whole if all notes start at the same time, else broken
    start_times = [note.start for note in notes]
    return 0 if len(set(start_times)) == 1 else 1

# Model Definition
class TinyHarmonyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_roots = nn.Embedding(128, 128)
        self.embed_types = nn.Embedding(CHORD_VOCAB_SIZE, 128)
        self.embed_key = nn.Embedding(KEY_VOCAB_SIZE, 32)
        self.embed_mode = nn.Embedding(MODE_VOCAB_SIZE, 16)
        self.embed_melody = nn.Embedding(128, 64)

        encoder_layer = nn.TransformerEncoderLayer(d_model=368, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.root_head = nn.Linear(368, 128)
        self.type_head = nn.Linear(368, CHORD_VOCAB_SIZE)
        self.style_head = nn.Linear(368, 2)  # Whole/Broken

    def forward(self, root_seq, type_seq, melody_seq, key_sig, mode_sig):
        root_embed = self.embed_roots(root_seq)
        type_embed = self.embed_types(type_seq)
        melody_embed = self.embed_melody(melody_seq)
        key_embed = self.embed_key(key_sig).unsqueeze(1).expand(-1, root_seq.size(1), -1)
        mode_embed = self.embed_mode(mode_sig).unsqueeze(1).expand(-1, root_seq.size(1), -1)

        x = torch.cat((root_embed, type_embed, melody_embed, key_embed, mode_embed), dim=-1)
        x = self.transformer(x)

        return self.root_head(x), self.type_head(x), self.style_head(x)

# Real Dataset Loader
def build_harmony_dataset():
    meta = pd.read_csv(META_CSV)
    if COMPOSER:
        meta = meta[meta['composer'].str.contains(COMPOSER, na=False)]
    files = meta['midi_filename'].tolist()
    sequences = []

    for fname in tqdm(files):
        midi_path = os.path.join(MIDI_BASE, fname)
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                notes = instrument.notes
                if len(notes) < 36:
                    continue

                seed_notes = notes[:4]
                target_notes = notes[4:36]

                seed_roots = [note.pitch for note in seed_notes]
                seed_types = [detect_chord_type(seed_notes) for _ in range(4)]
                seed_melody = [note.pitch for note in seed_notes]

                target_roots = [note.pitch for note in target_notes]
                target_types = [detect_chord_type(target_notes[i:i+3]) for i in range(len(target_notes))]
                target_melody = [note.pitch for note in target_notes]
                target_styles = [detect_style(target_notes[i:i+3]) for i in range(len(target_notes))]

                sequences.append((seed_roots, seed_types, seed_melody, 0, 0, target_roots, target_types, target_melody, target_styles))
        except:
            continue

    return sequences

# Training Loop
def train_harmony():
    sequences = build_harmony_dataset()
    random.shuffle(sequences)
    split = int(0.8 * len(sequences))
    train_data = sequences[:split]

    model = TinyHarmonyTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn_root = nn.CrossEntropyLoss()
    loss_fn_type = nn.CrossEntropyLoss()
    loss_fn_style = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for idx, (seed_roots, seed_types, seed_melody, key_signature, mode_idx,
                  target_roots, target_types, target_melody, target_styles) in enumerate(tqdm(train_data)):

            input_roots = seed_roots + target_roots[:-1]
            input_types = seed_types + target_types[:-1]
            input_melody = seed_melody + target_melody[:-1]

            seq_len = len(input_roots)
            assert len(input_types) == seq_len and len(input_melody) == seq_len, "Input sequences must match length!"

            seed_roots_tensor = torch.tensor(input_roots, dtype=torch.long).unsqueeze(0).to(DEVICE)
            seed_types_tensor = torch.tensor(input_types, dtype=torch.long).unsqueeze(0).to(DEVICE)
            seed_melody_tensor = torch.tensor(input_melody, dtype=torch.long).unsqueeze(0).to(DEVICE)

            target_roots_tensor = torch.tensor(target_roots, dtype=torch.long).to(DEVICE)
            target_types_tensor = torch.tensor(target_types, dtype=torch.long).to(DEVICE)
            target_styles_tensor = torch.tensor(target_styles, dtype=torch.long).to(DEVICE)

            key_sig_tensor = torch.tensor([key_signature], dtype=torch.long).to(DEVICE)
            mode_sig_tensor = torch.tensor([mode_idx], dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred_roots, pred_types, pred_styles = model(
                    seed_roots_tensor, seed_types_tensor, seed_melody_tensor, key_sig_tensor, mode_sig_tensor
                )

                pred_roots = pred_roots[:, -target_roots_tensor.size(0):, :]
                pred_types = pred_types[:, -target_types_tensor.size(0):, :]
                pred_styles = pred_styles[:, -target_styles_tensor.size(0):, :]

                pred_roots = pred_roots.squeeze(0)
                pred_types = pred_types.squeeze(0)
                pred_styles = pred_styles.squeeze(0)

                loss_root = loss_fn_root(pred_roots, target_roots_tensor)
                loss_type = loss_fn_type(pred_types, target_types_tensor)
                loss_style = loss_fn_style(pred_styles, target_styles_tensor)

                total_batch_loss = loss_root + loss_type + loss_style

            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += total_batch_loss.item()

        print(f"Epoch {epoch} Harmony Loss: {total_loss/len(train_data):.4f}")

        if epoch % CHECKPOINT_INTERVAL == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            checkpoint_path = os.path.join(SAVE_DIR, f"Harmony_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)

if __name__ == "__main__":
    import argparse
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device Found")

    parser = argparse.ArgumentParser(description="Tiny Harmony Transformer Trainer")
    parser.add_argument("--meta_csv", type=str, default=META_CSV, help="Path to metadata CSV")
    parser.add_argument("--midi_base", type=str, default=MIDI_BASE, help="Base path to MIDI files")
    parser.add_argument("--composer", type=str, default=None, help="Composer name to filter by")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Checkpoint save directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL, help="Checkpoint save interval")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    META_CSV = args.meta_csv
    MIDI_BASE = args.midi_base
    COMPOSER = args.composer
    SAVE_DIR = args.save_dir
    EPOCHS = args.epochs
    CHECKPOINT_INTERVAL = args.checkpoint_interval
    BATCH_SIZE = args.batch_size

    train_harmony()
