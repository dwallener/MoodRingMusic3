import os
import sys
import json
import random
import torch
import torch.nn as nn
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

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
        probs = torch.softmax(logits[0, -1] / temperature, dim=0).detach()
        next_token = torch.multinomial(probs, 1).item()
        if next_token == 0:
            break
        sequence.append(next_token)
    return [idx_to_token[t] for t in sequence if t != 0]

def orchestrate(model_folder, output_file="composition.mid", song_structure_file="song_structure.json"):
    ticks_per_beat = 480
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    tempo = mido.bpm2tempo(120)

    models = {}
    for f in os.listdir(model_folder):
        if f.endswith(".model.pt"):
            base = f.replace(".model.pt", "")
            parts = base.split("_")
            if len(parts) == 2:
                spine, length = parts
                models.setdefault(spine, {})[length] = base

    with open(song_structure_file, "r") as f:
        sections = json.load(f)

    model_cache = {}
    global_time_cursor = 0

    for section in sections:
        bars = section["length_bars"]
        section_ticks = bars * 4 * ticks_per_beat
        active_spines = section.get("active_spines", list(models.keys()))
        temperature = section.get("temperature", 1.0)

        for spine in active_spines:
            track = MidiTrack()
            mid.tracks.append(track)
            track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
            track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
            track.append(Message('program_change', program=40, time=0))

            current_ticks = 0
            attempt_counter = 0
            MAX_ATTEMPTS = 100

            first_event = True
            while current_ticks < section_ticks and attempt_counter < MAX_ATTEMPTS:
                remaining_ticks = section_ticks - current_ticks
                available_lengths = list(models[spine].keys())
                phrase_length = random.choice(available_lengths)

                model_key = (spine, phrase_length)
                if model_key in model_cache:
                    model, token_to_idx, idx_to_token = model_cache[model_key]
                else:
                    model_base = models[spine][phrase_length]
                    model_file = os.path.join(model_folder, f"{model_base}.model.pt")
                    vocab_file = os.path.join(model_folder, f"{model_base}.vocab.json")
                    model, token_to_idx, idx_to_token = load_model_and_vocab(model_file, vocab_file)
                    model_cache[model_key] = (model, token_to_idx, idx_to_token)

                tokens = generate_phrase(model, token_to_idx, idx_to_token, max_length=32, temperature=temperature)
                phrase_ticks = sum(int(token.split("_")[1]) * (ticks_per_beat // 4) for token in tokens if "_" in token)

                if phrase_ticks == 0:
                    attempt_counter += 1
                    continue  # Try again

                if phrase_ticks > remaining_ticks:
                    truncated_tokens = []
                    accumulated_ticks = 0
                    for token in tokens:
                        if "_" not in token:
                            continue
                        dur = int(token.split("_")[1]) * (ticks_per_beat // 4)
                        if accumulated_ticks + dur > remaining_ticks:
                            break
                        truncated_tokens.append(token)
                        accumulated_ticks += dur
                    tokens = truncated_tokens
                    phrase_ticks = accumulated_ticks

                    if phrase_ticks == 0:
                        attempt_counter += 1
                        continue

                local_time_cursor = 0
                for token in tokens:
                    if "_" not in token:
                        continue
                    pitch_midi, dur_sixteenths = map(int, token.split("_"))
                    duration = dur_sixteenths * (ticks_per_beat // 4)
                    if first_event:
                        time_offset = global_time_cursor
                        first_event = False
                    else:
                        time_offset = 0
                    track.append(Message('note_on', note=pitch_midi, velocity=64, time=time_offset))
                    track.append(Message('note_off', note=pitch_midi, velocity=64, time=duration))
                    local_time_cursor += duration

                current_ticks += phrase_ticks
                attempt_counter = 0  # Reset after successful addition

        global_time_cursor += section_ticks

    mid.save(output_file)
    print(f"[Success] MIDI composition exported to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrate.py <model_folder> [output_file] [song_structure.json]")
    else:
        model_folder = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "composition.mid"
        song_structure_file = sys.argv[3] if len(sys.argv) > 3 else "song_structure.json"
        orchestrate(model_folder, output_file, song_structure_file)
        