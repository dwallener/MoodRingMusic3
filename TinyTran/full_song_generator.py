import torch
import pretty_midi
import subprocess
import random
import json
import argparse
from composer_melody_trainer import TinyMelodyTransformer, INTERVAL_VOCAB, DURATION_VOCAB, DEVICE, build_dataset
from composer_harmony_trainer import TinyHarmonyTransformer, HARMONY_ROOT_VOCAB, HARMONY_TYPE_VOCAB, HARMONY_STYLE_VOCAB

# --- CLI Argument Parser ---
parser = argparse.ArgumentParser(description="Generate Full Song with Melody and Harmony")
parser.add_argument("--structure_file", type=str, required=True, help="Path to JSON song structure")
parser.add_argument("--melody_checkpoint", type=str, required=True, help="Path to melody model checkpoint")
parser.add_argument("--harmony_checkpoint", type=str, required=True, help="Path to harmony model checkpoint")
parser.add_argument("--soundfont", type=str, required=True, help="Path to SoundFont (.sf2) file")
parser.add_argument("--output_midi", type=str, default="full_song.mid", help="Output MIDI filename")
parser.add_argument("--bias_strength", type=float, default=0.05, help="Register gravity strength (default: 0.05)")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
args = parser.parse_args()

# --- Load Song Structure ---
with open(args.structure_file, 'r') as f:
    song_structure = json.load(f)

# --- Load Models ---
melody_model = TinyMelodyTransformer().to(DEVICE)
melody_ckpt = torch.load(args.melody_checkpoint, map_location=DEVICE)
melody_model.load_state_dict(melody_ckpt["model_state_dict"])
melody_model.eval()

harmony_model = TinyHarmonyTransformer().to(DEVICE)
harmony_ckpt = torch.load(args.harmony_checkpoint, map_location=DEVICE)
harmony_model.load_state_dict(harmony_ckpt["model_state_dict"])
harmony_model.eval()

# --- Prepare Dataset for Seeding ---
sequences = build_dataset()
split = int(0.8 * len(sequences))
val_data = sequences[split:]

final_melody_pitches, final_melody_durations = [], []
final_chord_roots, final_chord_types, final_chord_styles = [], [], []

broken_chord_patterns = [
    [0, 4, 7],           # Standard arpeggio
    [0, 7, 4],           # Reverse arpeggio
    [0, 4, 7, 12],       # Add octave
    [0, 3, 7],           # Minor feel
    [0, 5, 7],           # Suspended feel
]

def pick_broken_pattern():
    return random.choice(broken_chord_patterns)

for section in song_structure:
    section_name = section.get("section", "Section")
    bars = section.get("length_bars", 8)
    temperature = section.get("temperature", args.temperature)

    print(f"Generating Section: {section_name} | Bars: {bars}")

    seed, *_ = random.choice(val_data)
    generated = seed[:]
    durations = [0.5] * len(seed)

    input_seq = torch.tensor([INTERVAL_VOCAB.index(i) for i in seed], dtype=torch.long).unsqueeze(1).to(DEVICE)
    current_pitch = 60
    center_pitch = 60

    melody_pitches = []
    melody_durations = []

    target_beats = bars * 4
    current_duration = sum(durations)

    while current_duration < target_beats:
        with torch.no_grad():
            pred_intervals, pred_durations, _ = melody_model(input_seq)

            next_token_logits = pred_intervals[-1] / temperature
            current_bias = -args.bias_strength * (current_pitch - center_pitch)
            next_token_logits += current_bias
            next_token = torch.argmax(next_token_logits).item()
            next_interval = INTERVAL_VOCAB[next_token] if next_token < len(INTERVAL_VOCAB) else 0
            generated.append(next_interval)

            next_duration_logits = pred_durations[-1]
            next_duration_token = torch.argmax(next_duration_logits).item()
            next_duration = DURATION_VOCAB[next_duration_token]
            durations.append(next_duration)
            current_duration += next_duration

            input_seq = torch.cat([input_seq, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=0)

            current_pitch += next_interval
            current_pitch = max(0, min(127, current_pitch))
            melody_pitches.append(current_pitch)
            melody_durations.append(next_duration)

    final_melody_pitches.extend(melody_pitches)
    final_melody_durations.extend(melody_durations)

    # --- Generate Harmony Based on Melody ---
    melody_tensor = torch.tensor(melody_pitches, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_roots, pred_types, pred_styles = harmony_model(melody_tensor)
        chord_roots = torch.argmax(pred_roots, dim=-1).squeeze(0).cpu().tolist()
        chord_types = torch.argmax(pred_types, dim=-1).squeeze(0).cpu().tolist()
        chord_styles = torch.argmax(pred_styles, dim=-1).squeeze(0).cpu().tolist()

    final_chord_roots.extend(chord_roots)
    final_chord_types.extend(chord_types)
    final_chord_styles.extend(chord_styles)

# --- Create PrettyMIDI Song ---
midi = pretty_midi.PrettyMIDI()
melody_instr = pretty_midi.Instrument(program=0)  # Piano
harmony_instr = pretty_midi.Instrument(program=48)  # Strings

start_time = 0.0
for pitch, duration in zip(final_melody_pitches, final_melody_durations):
    note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration)
    melody_instr.notes.append(note)
    start_time += duration

# --- Add Harmony Based on Melody-Driven Timing ---
harmony_time = 0.0
for i, (root, ctype, style) in enumerate(zip(final_chord_roots, final_chord_types, final_chord_styles)):
    root_pitch = HARMONY_ROOT_VOCAB[root] if root < len(HARMONY_ROOT_VOCAB) else 60
    if i < len(final_melody_durations):
        duration = final_melody_durations[i]  # Match harmony to melody duration
    else:
        duration = 1.0

    if style == 0:  # Whole chord
        chord_pitches = [root_pitch, root_pitch + 4, root_pitch + 7]
        for cp in chord_pitches:
            note = pretty_midi.Note(velocity=80, pitch=int(cp), start=harmony_time, end=harmony_time + duration)
            harmony_instr.notes.append(note)
    else:  # Broken chord (arpeggiated)
        pattern = pick_broken_pattern()
        step_duration = duration / len(pattern)
        for j, interval in enumerate(pattern):
            note_start = harmony_time + j * step_duration
            note = pretty_midi.Note(velocity=80, pitch=int(root_pitch + interval), start=note_start, end=note_start + step_duration)
            harmony_instr.notes.append(note)
    harmony_time += duration

midi.instruments.append(melody_instr)
midi.instruments.append(harmony_instr)
midi.write(args.output_midi)

# --- Play Final Song ---
subprocess.run(["fluidsynth", "-ni", args.soundfont, args.output_midi, "-F", "out.wav", "-r", "44100"])

try:
    subprocess.run(["open", "out.wav"])  # MacOS
except FileNotFoundError:
    subprocess.run(["aplay", "out.wav"])  # Linux Alternative
    