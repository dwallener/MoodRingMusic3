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

def diatonically_adjust_phrase(phrase_tokens, target_key, mode="major"):
    # Define semitone offsets for major and minor scales
    scale_degrees = {
        "major": [0, 2, 4, 5, 7, 9, 11],  # Ionian
        "minor": [0, 2, 3, 5, 7, 8, 10]   # Aeolian (Natural Minor)
    }

    if mode not in scale_degrees:
        raise ValueError(f"Unsupported mode: {mode}")

    # MIDI note numbers for C = 0, C# = 1, ..., B = 11
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_base_note = note_names.index(target_key)

    adjusted_tokens = []

    for token in phrase_tokens:
        if "_" not in token:
            adjusted_tokens.append(token)
            continue

        pitch_midi, dur_sixteenths = map(int, token.split("_"))

        # Calculate scale degree position
        midi_note_in_octave = pitch_midi % 12
        octave = pitch_midi // 12

        # Find closest scale degree in target key
        target_scale = [(key_base_note + interval) % 12 for interval in scale_degrees[mode]]

        # Find the nearest note in the target scale
        closest_note = min(target_scale, key=lambda n: abs(n - midi_note_in_octave))

        # Recalculate adjusted pitch
        adjusted_pitch = octave * 12 + closest_note

        adjusted_tokens.append(f"{adjusted_pitch}_{dur_sixteenths}")

    return adjusted_tokens


def analyze_energy(mid, sections, ticks_per_beat=480):
    section_starts = []
    current_tick = 0
    for section in sections:
        bars = section["length_bars"]
        section_starts.append((current_tick, section))
        current_tick += bars * 4 * ticks_per_beat

    print("\n[Energy Analysis]")
    section_summaries = {}

    for track_idx, track in enumerate(mid.tracks):
        current_section_idx = 0
        note_ons = {}
        current_tick = 0

        for msg in track:
            current_tick += msg.time
            while (current_section_idx + 1 < len(section_starts) and
                   current_tick >= section_starts[current_section_idx + 1][0]):
                current_section_idx += 1

            section_start_tick, section = section_starts[current_section_idx]
            section_name = section["section"]
            target_energy = float(section.get("energy", 1.0))

            if msg.type == 'note_on' and msg.velocity > 0:
                note_ons.setdefault(section_name, 0)
                note_ons[section_name] += 1
                section_summaries.setdefault(section_name, []).append((track_idx, section.get("energy", 1.0)))

        for section_name, total_note_ons in note_ons.items():
            bars = next(sec["length_bars"] for sec in sections if sec["section"] == section_name)
            measured_notes_per_measure = total_note_ons / bars
            target_energy = next(sec.get("energy", 1.0) for sec in sections if sec["section"] == section_name)
            target_notes_per_measure = target_energy * 8
            print(f"  Track {track_idx:2d} | Section: {section_name:12s} | "
                  f"Target: {target_notes_per_measure:5.1f} | "
                  f"Measured: {measured_notes_per_measure:5.1f}")

    # Output total per section
    for section_name in section_summaries:
        bars = next(sec["length_bars"] for sec in sections if sec["section"] == section_name)
        target_energy = next(sec.get("energy", 1.0) for sec in sections if sec["section"] == section_name)
        target_notes_per_measure = target_energy * 8

        total_note_ons = 0
        for track in mid.tracks:
            current_tick = 0
            section_start_tick = next(tick for tick, sec in section_starts if sec["section"] == section_name)
            section_end_tick = section_start_tick + bars * 4 * ticks_per_beat

            for msg in track:
                current_tick += msg.time
                if section_start_tick <= current_tick < section_end_tick:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        total_note_ons += 1

        measured_notes_per_measure = total_note_ons / bars
        print(f"  >>> Total Energy for Section: {section_name:12s} | "
              f"Target: {target_notes_per_measure:5.1f} | "
              f"Measured: {measured_notes_per_measure:5.1f}\n")


def apply_energy_envelope(mid, sections, ticks_per_beat=480):
    # Precompute section start times in ticks
    section_starts = []
    current_tick = 0
    for section in sections:
        bars = section["length_bars"]
        section_starts.append((current_tick, section))
        current_tick += bars * 4 * ticks_per_beat

    for track in mid.tracks:
        new_messages = []
        pending_delta = 0
        current_tick = 0
        current_section_idx = 0

        for msg in track:
            current_tick += msg.time

            # Advance to the correct section based on current_tick
            while (current_section_idx + 1 < len(section_starts) and
                   current_tick >= section_starts[current_section_idx + 1][0]):
                current_section_idx += 1

            section_start_tick, section = section_starts[current_section_idx]
            bars = section["length_bars"]
            section_end_tick = section_start_tick + bars * 4 * ticks_per_beat
            target_energy = section.get("energy", 1.0) * 8  # Target notes per measure

            # Calculate measured energy within this section for this track
            temp_tick = 0
            section_note_count = 0
            for m in track:
                temp_tick += m.time
                if section_start_tick <= temp_tick < section_end_tick:
                    if m.type == 'note_on' and m.velocity > 0:
                        section_note_count += 1

            measured_energy = section_note_count / bars if bars else 0

            # Apply the energy selection logic
            keep = False
            if measured_energy < 3:
                keep = True
            elif measured_energy <= target_energy * 1.25:
                keep = True

            in_current_section = section_start_tick <= current_tick < section_end_tick

            if keep or not in_current_section:
                # Keep this event, apply pending delta time
                msg.time += pending_delta
                new_messages.append(msg)
                pending_delta = 0
            else:
                # Skip this event, accumulate its time
                pending_delta += msg.time

        # Replace track events
        track.clear()
        track.extend(new_messages)

def select_tracks_for_energy(target_energy, tracks_with_energy):
    """
    Select tracks whose combined energy falls within ±25% of the target energy.
    
    Args:
        target_energy (float): The target energy (notes per bar).
        tracks_with_energy (list of tuples): List of (track_index, measured_energy) pairs.

    Returns:
        list of int: Selected track indices.
    """
    tracks_with_energy.sort(key=lambda x: -x[1])  # Sort tracks by highest energy first
    selected_tracks = []
    total_energy = 0
    lower_bound = target_energy * 0.75
    upper_bound = target_energy * 1.25

    for track_idx, energy in tracks_with_energy:
        selected_tracks.append(track_idx)
        total_energy += energy
        if lower_bound <= total_energy <= upper_bound:
            break  # Target reached within acceptable window

    return selected_tracks

def analyze_energy_deltas(mid, sections, ticks_per_beat=480):
    section_starts = []
    current_tick = 0
    for section in sections:
        bars = section["length_bars"]
        section_starts.append((current_tick, section))
        current_tick += bars * 4 * ticks_per_beat

    print("\n[Energy Delta Analysis]")
    section_note_counts = {sec["section"]: 0 for sec in sections}

    for track in mid.tracks:
        current_tick = 0
        current_section_idx = 0

        for msg in track:
            current_tick += msg.time
            while (current_section_idx + 1 < len(section_starts) and
                   current_tick >= section_starts[current_section_idx + 1][0]):
                current_section_idx += 1

            section_start_tick, section = section_starts[current_section_idx]
            section_name = section["section"]
            if msg.type == 'note_on' and msg.velocity > 0:
                section_note_counts[section_name] += 1

    for section in sections:
        name = section["section"]
        bars = section["length_bars"]
        target_energy = section.get("energy", 1.0)
        target_notes_per_bar = target_energy * 8
        measured_notes_per_bar = section_note_counts[name] / bars
        delta = measured_notes_per_bar - target_notes_per_bar

        print(f"  Section: {name:12s} | Target: {target_notes_per_bar:5.1f} | "
              f"Measured: {measured_notes_per_bar:5.1f} | "
              f"Delta: {delta:+5.1f} notes/bar")


def analyze_energy_deltas_with_selection(mid, sections, ticks_per_beat=480):
    section_starts = []
    current_tick = 0
    for section in sections:
        bars = section["length_bars"]
        section_starts.append((current_tick, section))
        current_tick += bars * 4 * ticks_per_beat

    print("\n[Energy Delta Analysis with Track Selection]")
    section_track_energy = {}

    for track_idx, track in enumerate(mid.tracks):
        current_tick = 0
        current_section_idx = 0

        for msg in track:
            current_tick += msg.time
            while (current_section_idx + 1 < len(section_starts) and
                   current_tick >= section_starts[current_section_idx + 1][0]):
                current_section_idx += 1

            section_start_tick, section = section_starts[current_section_idx]
            section_name = section["section"]
            section_track_energy.setdefault(section_name, {}).setdefault(track_idx, 0)

            if msg.type == 'note_on' and msg.velocity > 0:
                section_track_energy[section_name][track_idx] += 1

    for section in sections:
        name = section["section"]
        bars = section["length_bars"]
        target_energy = section.get("energy", 1.0)
        target_notes_per_bar = target_energy * 8

        track_energies = []
        track_energy_data = section_track_energy.get(name, {})
        for track_idx, note_count in track_energy_data.items():
            energy_per_bar = note_count / bars
            track_energies.append((track_idx, energy_per_bar))

        selected_tracks = select_tracks_for_energy(target_notes_per_bar, track_energies)

        before_tracks = sorted(track_energy_data.keys())
        after_tracks = sorted(selected_tracks)

        before_str = ", ".join(str(t) for t in before_tracks)
        after_str = ", ".join(str(t) for t in after_tracks)

        print(f"  Section: {name:12s} | Target: {target_notes_per_bar:5.1f} notes/bar")
        print(f"    Before: {before_str}")
        print(f"    After:  {after_str}\n")
        

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
                tokens = diatonically_adjust_phrase(tokens, section["key"], section["mode"])
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

    analyze_energy(mid, sections, ticks_per_beat)    
    mid.save(output_file)
    print(f"[Success] MIDI composition exported to {output_file}")

    analyze_energy_deltas(mid, sections, ticks_per_beat)
    analyze_energy_deltas_with_selection(mid, sections, ticks_per_beat)


    #apply_energy_envelope(mid, sections, ticks_per_beat)
    #analyze_energy(mid, sections, ticks_per_beat)
    #mid.save(output_file_energy)
    #print(f"[Success] MIDI composition --pruned-- exported to {output_file_energy}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrate.py <model_folder> [output_file] [song_structure.json]")
    else:
        model_folder = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "composition.mid"
        output_file_energy = "pruned_" + output_file 
        song_structure_file = sys.argv[3] if len(sys.argv) > 3 else "song_structure.json"
        orchestrate(model_folder, output_file, song_structure_file)
