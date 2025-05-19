import os
import json
import random
import sys
import torch
import torch.nn as nn
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# -----------------------
# Tiny Transformer Model
# -----------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embed(x).permute(1, 0, 2)
        out = self.transformer(emb).permute(1, 0, 2)
        logits = self.fc_out(out)
        return logits

# -----------------------
# Utility Functions
# -----------------------
def load_model_and_vocab(model_file, vocab_file):
    with open(vocab_file, "r") as f:
        vocab_data = json.load(f)
    token_to_idx = vocab_data["token_to_idx"]
    idx_to_token = {int(k): v for k, v in vocab_data["idx_to_token"].items()}

    model = TinyTransformer(len(token_to_idx))
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model, token_to_idx, idx_to_token

def generate_phrase(model, token_to_idx, idx_to_token, max_length, temperature=1.0):
    sequence = [random.choice(list(token_to_idx.values()))]
    for _ in range(max_length - 1):
        input_seq = torch.tensor(sequence).unsqueeze(0)
        logits = model(input_seq)
        probs = torch.softmax(logits[0, -1] / temperature, dim=0).detach().numpy()
        next_token = random.choices(range(len(probs)), weights=probs, k=1)[0]
        if next_token == 0:
            break
        sequence.append(next_token)
    return [idx_to_token[t] for t in sequence if t != 0]

def infer_chord(bass_notes):
    return max(set(bass_notes), key=bass_notes.count) if bass_notes else 60  # Default to C if no bass notes found

def get_allowed_pitches(key, mode):
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]
    key_offsets = {k: i for i, k in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
    base_offset = key_offsets.get(key.upper(), 0)
    scale = major_scale if mode.lower() == "major" else minor_scale
    return [(note + base_offset) % 12 for note in scale]

def transpose_pitch(pitch, target_key, mode):
    key_offsets = {k: i for i, k in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
    return pitch + key_offsets.get(target_key.upper(), 0)

def orchestrate(model_folder, output_file="composition_mido.mid", song_structure_file="song_structure.json"):
    ticks_per_beat = 480
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    tempo = mido.bpm2tempo(120)

    # Auto-detect available spines from model folder
    models = {}
    for f in os.listdir(model_folder):
        if f.endswith(".model.pt"):
            base = f.replace(".model.pt", "")
            parts = base.split("_")
            if len(parts) == 2:
                spine, length = parts
                models.setdefault(spine, {})[length] = base

    spine_names = sorted(models.keys())
    print(f"[Info] Detected Spines: {spine_names}")

    # Load song structure
    with open(song_structure_file, "r") as f:
        sections = json.load(f)

    # Track per spine
    for spine in spine_names:
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        track.append(Message('program_change', program=40, time=0))  # Default Violin patch

        for section in sections:
            bars_remaining = section["length_bars"]
            temperature = section.get("temperature", 1.0)
            key = section.get("key", "C")
            mode = section.get("mode", "major")
            allowed_pitches = get_allowed_pitches(key, mode)
            time_cursor = 0

            while bars_remaining > 0:
                # Randomly choose available length for this spine
                available_lengths = list(models[spine].keys())
                phrase_length = random.choice(available_lengths)
                model_base = models[spine][phrase_length]
                
                model_file = os.path.join(model_folder, f"{model_base}.model.pt")
                vocab_file = os.path.join(model_folder, f"{model_base}.vocab.json")
                
                model, token_to_idx, idx_to_token = load_model_and_vocab(model_file, vocab_file)
                tokens = generate_phrase(model, token_to_idx, idx_to_token, max_length=32, temperature=temperature)

                for token in tokens:
                    if "_" not in token or token in ["=", "rest", "."]:
                        continue  # Skip measure markers and rests

                    try:
                        pitch_midi, dur_sixteenths = map(int, token.split("_"))
                    except ValueError:
                        continue  # Skip malformed tokens safely

                    if pitch_midi % 12 not in allowed_pitches:
                        continue  # Skip out-of-key notes

                    pitch_midi = transpose_pitch(pitch_midi, key, mode)
                    duration = dur_sixteenths * int(ticks_per_beat / 4)
                    track.append(Message('note_on', note=pitch_midi, velocity=64, time=time_cursor))
                    track.append(Message('note_off', note=pitch_midi, velocity=64, time=duration))
                    time_cursor = 0

                bars_remaining -= 2  # Rough estimate for each phrase

    mid.save(output_file)
    print(f"[Success] MIDI composition exported to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrate_on_phrase_dataset.py <model_folder> [output_file]")
    else:
        model_folder = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "composition_mido.mid"
        orchestrate(model_folder, output_file)


