import torch
import torch.nn as nn
import json
import random
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
        emb = self.embed(x)
        emb = emb.permute(1, 0, 2)
        out = self.transformer(emb)
        out = out.permute(1, 0, 2)
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

def load_transition_matrix(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def choose_next_phrase(current_state, matrix):
    transitions = matrix.get(current_state, {})
    if not transitions:
        return "Silent"
    choices, weights = zip(*transitions.items())
    return random.choices(choices, weights=weights, k=1)[0]

def transpose_pitch(pitch, target_key, mode):
    key_offsets = {
        "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
        "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
    }
    offset = key_offsets.get(target_key.upper(), 0)
    return pitch + offset

def register_shift(pitch, register, instrument):
    base_shifts = {"Cello": -24, "Viola": -12, "Violin1": 0, "Violin2": 0}
    shift = base_shifts.get(instrument, 0)

    if register == "low":
        shift -= 12
    elif register == "high":
        shift += 12

    return max(0, min(127, pitch + shift))

# -----------------------
# Orchestration Logic
# -----------------------
def orchestrate(output_file="composition_mido.mid", 
                transition_matrix_file="cello_transition_matrix.json", 
                song_structure_file="song_structure.json"):

    instruments = {
        "Cello": 32,    # Acoustic Bass
        "Viola": 41,    # Violin patch for viola proxy
        "Violin1": 40,  # Violin
        "Violin2": 40
    }

    phrase_lengths = {"Short": 8, "Medium": 16, "Long": 32}
    ticks_per_beat = 480
    matrix = load_transition_matrix(transition_matrix_file)

    with open(song_structure_file, "r") as f:
        sections = json.load(f)

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    tempo = mido.bpm2tempo(120)

    for track_name, program in instruments.items():
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        track.append(Message('program_change', program=program, time=0))

        for section in sections:
            bars_remaining = section["length_bars"]
            temperature = section["temperature"]
            register = section["register"]
            key = section["key"]
            mode = section["mode"]

            current_state = "Short"
            time_cursor = 0

            while bars_remaining > 0:
                phrase_type = choose_next_phrase(current_state, matrix)
                current_state = phrase_type

                if phrase_type == "Silent":
                    rest_ticks = 4 * ticks_per_beat
                    track.append(Message('note_off', time=rest_ticks))
                    bars_remaining -= 1
                    continue

                model_prefix = f"{track_name.lower()}_{phrase_type.lower()}"
                model_file = f"{model_prefix}.model.pt"
                vocab_file = f"{model_prefix}.vocab.json"

                model, token_to_idx, idx_to_token = load_model_and_vocab(model_file, vocab_file)
                tokens = generate_phrase(model, token_to_idx, idx_to_token, 
                                         max_length=phrase_lengths[phrase_type], 
                                         temperature=temperature)

                total_phrase_quarters = 0
                for token in tokens:
                    pitch_midi, dur_sixteenths = map(int, token.split("_"))
                    pitch_midi = transpose_pitch(pitch_midi, key, mode)
                    pitch_midi = register_shift(pitch_midi, register, track_name)
                    duration = dur_sixteenths * int(ticks_per_beat / 4)

                    track.append(Message('note_on', note=pitch_midi, velocity=64, time=time_cursor))
                    track.append(Message('note_off', note=pitch_midi, velocity=64, time=duration))

                    time_cursor = 0
                    total_phrase_quarters += dur_sixteenths / 4

                bars_remaining -= total_phrase_quarters / 4
                # Pad with rest if overshoot
                if bars_remaining < 0:
                    overshoot_ticks = int(abs(bars_remaining) * 4 * ticks_per_beat)
                    track.append(Message('note_off', time=overshoot_ticks))
                    bars_remaining = 0

    mid.save(output_file)
    print(f"MIDI composition exported to {output_file}")

# -----------------------
# Run Example
# -----------------------
if __name__ == "__main__":
    orchestrate()