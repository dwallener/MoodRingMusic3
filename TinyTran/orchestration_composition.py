import torch
import torch.nn as nn
import json
import sys
import random
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# -----------------------
# Model Definition
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
        next_token = random.choices(list(range(len(probs))), weights=probs, k=1)[0]
        if next_token == 0:
            break
        sequence.append(next_token)
    return [idx_to_token[t] for t in sequence if t != 0]

# -----------------------
# Orchestration Logic
# -----------------------
def orchestrate(structure_sequence, output_file="composition_mido.mid"):
    model_files = {
        "Short": ("tiny_transformer_phrase_dataset_short.json.pt", "tiny_transformer_phrase_dataset_short.vocab.json"),
        "Medium": ("tiny_transformer_phrase_dataset_medium.json.pt", "tiny_transformer_phrase_dataset_medium.vocab.json"),
        "Long": ("tiny_transformer_phrase_dataset_long.json.pt", "tiny_transformer_phrase_dataset_long.vocab.json")
    }

    ticks_per_beat = 480
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track.append(Message('program_change', program=0, time=0))

    for phrase_type in structure_sequence:
        if phrase_type == "Silent":
            rest_ticks = 4 * ticks_per_beat
            track.append(Message('note_off', time=rest_ticks))  # Just move the timeline forward
            continue

        model_file, vocab_file = model_files[phrase_type]
        model, token_to_idx, idx_to_token = load_model_and_vocab(model_file, vocab_file)

        phrase_length = {"Short": 8, "Medium": 16, "Long": 32}[phrase_type]
        tokens = generate_phrase(model, token_to_idx, idx_to_token, max_length=phrase_length)

        active_note = None

        for token in tokens:
            pitch_midi, dur_sixteenths = map(int, token.split("_"))
            duration = dur_sixteenths * 120  # assuming 120 ticks per quarter note

            # Explicitly turn off the last note before starting the next
            if active_note is not None:
                track.append(mido.Message('note_off', note=active_note, velocity=64, time=0))

            # Start the new note
            track.append(mido.Message('note_on', note=pitch_midi, velocity=64, time=0))
            track.append(mido.Message('note_off', note=pitch_midi, velocity=64, time=duration))

            active_note = pitch_midi
    
    mid.save(output_file)
    print(f"MIDI composition exported to {output_file}")

# -----------------------
# Example Usage
# -----------------------
if __name__ == "__main__":
    # Example structure; replace with generated structure if desired
    example_structure = ["Short", "Medium", "Silent", "Short", "Long", "Short", "Medium", "Silent", "Long"]
    orchestrate(example_structure, output_file="generated_composition_mido.mid")