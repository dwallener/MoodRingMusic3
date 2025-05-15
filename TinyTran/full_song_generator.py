import torch
import pretty_midi
import subprocess
import random
import json
import argparse
from composer_melody_trainer import (
    TinyMelodyTransformer, INTERVAL_VOCAB, DURATION_VOCAB, DEVICE, build_dataset
)

# CLI Argument Parser
parser = argparse.ArgumentParser(description="Test Trained Melody Generator with Register Bias")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--soundfont", type=str, required=True, help="Path to SoundFont (.sf2) file")
parser.add_argument("--output_midi", type=str, default="generated_melody.mid", help="Output MIDI filename")
parser.add_argument("--bias_strength", type=float, default=0.05, help="Register gravity strength (default: 0.05)")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
parser.add_argument("--structure_file", type=str, required=True, help="Path to JSON song structure")
args = parser.parse_args()

# Load Song Structure
with open(args.structure_file, 'r') as f:
    song_structure = json.load(f)

# Load Model
model = TinyMelodyTransformer().to(DEVICE)
checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Prepare Dataset for Seeding
sequences = build_dataset()
split = int(0.8 * len(sequences))
val_data = sequences[split:]

# Generate Song Sections
final_pitches = []
final_durations = []

for section in song_structure:
    section_name = section.get("section", "Section")
    bars = section.get("length_bars", 8)
    temperature = section.get("temperature", 0.8)
    register = section.get("register", "mid")
    key = section.get("key", "C")
    mode = section.get("mode", "major")

    print(f"Generating Section: {section_name} | Bars: {bars} | Key: {key} {mode} | Register: {register}")

    seed, *_ = random.choice(val_data)
    generated = seed[:]
    durations = [0.5] * len(seed)  # Default duration for seed notes

    input_seq = torch.tensor(
        [INTERVAL_VOCAB.index(i) for i in seed], dtype=torch.long
    ).unsqueeze(1).to(DEVICE)

    current_pitch = 60  # Middle C
    pitches = []

    center_pitch = 60

    for _ in range(bars * 8):  # Assuming 8 notes per bar
        with torch.no_grad():
            pred_intervals, pred_durations, pred_registers = model(input_seq)

            # Apply Temperature and Register Bias
            next_token_logits = pred_intervals[-1] / args.temperature
            current_bias = -args.bias_strength * (current_pitch - center_pitch)
            next_token_logits += current_bias

            next_token = torch.argmax(next_token_logits).item()
            next_interval = INTERVAL_VOCAB[next_token] if next_token < len(INTERVAL_VOCAB) else 0
            generated.append(next_interval)

            # Predict Duration as before
            next_duration_logits = pred_durations[-1]
            next_duration_token = torch.argmax(next_duration_logits).item()
            next_duration = DURATION_VOCAB[next_duration_token]
            durations.append(next_duration)

            input_seq = torch.cat(
                [input_seq, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=0
            )

            # Update pitch and clamp to valid range
            current_pitch += next_interval
            current_pitch = max(0, min(127, current_pitch))
            pitches.append(current_pitch)

    final_pitches.extend(pitches)
    final_durations.extend(durations)

# Create PrettyMIDI Song
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
start_time = 0.0

for pitch, duration in zip(final_pitches, final_durations):
    note = pretty_midi.Note(
        velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration
    )
    instrument.notes.append(note)
    start_time += duration

midi.instruments.append(instrument)
midi.write(args.output_midi)

# Play the Final Song
subprocess.run(["fluidsynth", "-ni", args.soundfont, args.output_midi, "-F", "out.wav", "-r", "44100"])

# Auto-play depending on OS (adjust as needed)
try:
    subprocess.run(["open", "out.wav"])  # MacOS
except FileNotFoundError:
    subprocess.run(["aplay", "out.wav"])  # Linux alternative