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

def generate_phrase(model, token_to_idx, idx_to_token, max_length):
    sequence = [random.choice(list(token_to_idx.values()))]
    for _ in range(max_length - 1):
        input_seq = torch.tensor(sequence).unsqueeze(0)
        logits = model(input_seq)
        probs = torch.softmax(logits[0, -1], dim=0).detach().numpy()
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

# -----------------------
# Orchestration Logic
# -----------------------
def orchestrate(output_file="composition_mido.mid", transition_matrix_file="cello_transition_matrix.json", total_phrases=32):
    # Model and vocab files per instrument
    instruments = {
        "Cello": 42,    # MIDI Program: Acoustic Bass
        "Viola": 41,    # MIDI Program: Violin (using as proxy for Viola)
        "Violin1": 40,  # MIDI Program: Violin
        "Violin2": 40
    }

    register_offsets = {
        "Cello": -24,    # Two octaves down
        "Viola": -12,    # One octave down
        "Violin1": 0,    # No change
        "Violin2": 0
    }

    phrase_lengths = {"Short": 8, "Medium": 16, "Long": 32}
    ticks_per_beat = 480
    matrix = load_transition_matrix(transition_matrix_file)

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    tempo = mido.bpm2tempo(120)

    for track_name, program in instruments.items():
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        track.append(Message('program_change', program=program, time=0))

        current_state = "Short"
        time_cursor = 0

        for _ in range(total_phrases):
            phrase_type = choose_next_phrase(current_state, matrix)
            current_state = phrase_type

            if phrase_type == "Silent":
                rest_ticks = 4 * ticks_per_beat
                track.append(Message('note_off', time=rest_ticks))
                continue

            model_prefix = f"{track_name.lower()}_{phrase_type.lower()}"
            model_file = f"{model_prefix}.model.pt"
            vocab_file = f"{model_prefix}.vocab.json"

            model, token_to_idx, idx_to_token = load_model_and_vocab(model_file, vocab_file)
            tokens = generate_phrase(model, token_to_idx, idx_to_token, max_length=phrase_lengths[phrase_type])

            for token in tokens:
                #pitch_midi, dur_sixteenths = map(int, token.split("_"))
                pitch_midi, dur_sixteenths = map(int, token.split("_"))
                pitch_midi += register_offsets[track_name]  # Shift pitch based on instrument
                duration = dur_sixteenths * int(ticks_per_beat / 4)
                track.append(Message('note_on', note=pitch_midi, velocity=64, time=time_cursor))
                track.append(Message('note_off', note=pitch_midi, velocity=64, time=duration))
                time_cursor = 0

    mid.save(output_file)
    print(f"MIDI composition exported to {output_file}")

# -----------------------
# Run Example
# -----------------------
if __name__ == "__main__":
    orchestrate()
