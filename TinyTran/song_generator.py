import torch
import pretty_midi
import subprocess
import random
import json
import argparse
import numpy as np
from composer_melody_trainer import (
    TinyMelodyTransformer, INTERVAL_VOCAB, DURATION_VOCAB, DEVICE, build_dataset
)
from composer_harmony_trainer import TinyHarmonyTransformer, HARMONY_ROOT_VOCAB, HARMONY_TYPE_VOCAB, DEVICE as DEVICE_HARMONY

# --- Mappings ---
BPM_RANGES = {
    "Focus": {"Morning": (90, 110), "Midday": (100, 120), "Evening": (80, 100), "Night": (60, 80)},
    "Relax": {"Morning": (70, 90), "Midday": (80, 100), "Evening": (60, 80), "Night": (40, 60)},
    "Energy": {"Morning": (100, 120), "Midday": (120, 140), "Evening": (110, 130), "Night": (90, 110)},
    "Sleep": {"Evening": (50, 70), "Night": (40, 60)},
    "Creative Flow": {"Morning": (90, 110), "Midday": (100, 120), "Evening": (80, 100)},
    "Calm Confidence": {"Morning": (80, 100), "Midday": (90, 110), "Evening": (70, 90)},
    "Romantic": {"Evening": (60, 80), "Night": (50, 70)},
    "Reflective": {"Evening": (60, 80), "Night": (50, 70)}
}

KEY_MAP = {
    "Focus": ["C", "A", "F"], "Relax": ["F", "Eb", "Bb"], "Energy": ["G", "D", "E"],
    "Sleep": ["A", "C", "G"], "Creative Flow": ["E", "G", "D"], "Calm Confidence": ["C", "F", "Bb"],
    "Romantic": ["Bb", "D", "A"], "Reflective": ["C", "D", "Eb", "G"]
}

SCALE_NOTES = {
    "C": [0, 2, 4, 5, 7, 9, 11], "D": [2, 4, 6, 7, 9, 11, 1], "E": [4, 6, 8, 9, 11, 1, 3],
    "F": [5, 7, 9, 10, 0, 2, 4], "G": [7, 9, 11, 0, 2, 4, 6], "A": [9, 11, 1, 2, 4, 6, 8],
    "Bb": [10, 0, 2, 3, 5, 7, 9], "Eb": [3, 5, 7, 8, 10, 0, 2]
}

# --- CLI Parser ---
parser = argparse.ArgumentParser(description="Full Song Generator with Mood and Key Control")
parser.add_argument("--structure_file", type=str, required=True)
parser.add_argument("--melody_checkpoint", type=str, required=True)
parser.add_argument("--harmony_checkpoint", type=str, required=True)
parser.add_argument("--soundfont", type=str, required=True)
parser.add_argument("--output_midi", type=str, default="full_song.mid")
parser.add_argument("--mood", type=str, required=True, choices=list(BPM_RANGES.keys()))
parser.add_argument("--circadian_phase", type=str, required=True, choices=["Morning", "Midday", "Evening", "Night"])
parser.add_argument("--bias_strength", type=float, default=0.05)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--melody_key_penalty", type=float, default=0.5)
parser.add_argument("--harmony_key_strictness", type=float, default=1.0)
args = parser.parse_args()

# --- Load Models ---
melody_model = TinyMelodyTransformer().to(DEVICE)
melody_checkpoint = torch.load(args.melody_checkpoint, map_location=DEVICE)
melody_model.load_state_dict(melody_checkpoint["model_state_dict"])
melody_model.eval()

harmony_model = TinyHarmonyTransformer().to(DEVICE_HARMONY)
harmony_checkpoint = torch.load(args.harmony_checkpoint, map_location=DEVICE_HARMONY)
harmony_model.load_state_dict(harmony_checkpoint["model_state_dict"])
harmony_model.eval()

# --- Seed Data ---
sequences = build_dataset()
split = int(0.8 * len(sequences))
val_data = sequences[split:]

# --- Mood and Key Selection ---
bpm = random.randint(*BPM_RANGES[args.mood][args.circadian_phase])
key = random.choice(KEY_MAP[args.mood])
scale_notes = SCALE_NOTES[key]

print(f"\n🎵 Generating in Key: {key} Major | BPM: {bpm} | Mood: {args.mood} | Phase: {args.circadian_phase}")

# --- Load Structure ---
with open(args.structure_file, 'r') as f:
    song_structure = json.load(f)

final_pitches, final_durations, final_chords = [], [], []

for section in song_structure:
    section_name = section.get("section", "Section")
    bars = section.get("length_bars", 8)
    print(f"🎶 Section: {section_name} | Bars: {bars}")

    seed, *_ = random.choice(val_data)
    generated = seed[:]
    durations = [0.5] * len(seed)

    input_seq = torch.tensor(
        [INTERVAL_VOCAB.index(i) for i in seed], dtype=torch.long
    ).unsqueeze(1).to(DEVICE)

    current_pitch = 60
    pitches = []
    center_pitch = 60

    target_beats = bars * 4
    current_duration = sum(durations)

    while current_duration < target_beats:
        with torch.no_grad():
            pred_intervals, pred_durations, _ = melody_model(input_seq)

            # Apply Melody Key Penalty
            logits = pred_intervals[-1].cpu().numpy()
            for idx, interval in enumerate(INTERVAL_VOCAB):
                test_pitch = (current_pitch + interval) % 12
                if test_pitch not in scale_notes:
                    logits[idx] -= args.melody_key_penalty

            next_token = np.argmax(logits)
            next_interval = INTERVAL_VOCAB[next_token] if next_token < len(INTERVAL_VOCAB) else 0
            generated.append(next_interval)

            # Duration
            next_duration_logits = pred_durations[-1]
            next_duration_token = torch.argmax(next_duration_logits).item()
            next_duration = DURATION_VOCAB[next_duration_token]
            durations.append(next_duration)
            current_duration += next_duration

            input_seq = torch.cat(
                [input_seq, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=0
            )

            current_pitch += next_interval
            current_pitch = max(0, min(127, current_pitch))
            pitches.append(current_pitch)

    # Harmony Generation (Placeholder: one chord per bar for simplicity)
    chords = []
    for _ in range(bars):
        root = random.choice(scale_notes)
        chord = (root, "maj")  # Simplified to major chords for now
        chords.append(chord)

    final_pitches.extend(pitches)
    final_durations.extend(durations)
    final_chords.extend(chords)

# --- MIDI Rendering ---
midi = pretty_midi.PrettyMIDI()
melody_instr = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
harmony_instr = pretty_midi.Instrument(program=48)  # Strings Ensemble

start_time = 0.0

for pitch, duration in zip(final_pitches, final_durations):
    note = pretty_midi.Note(
        velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration
    )
    melody_instr.notes.append(note)
    start_time += duration

# Simple harmony rendering (pad-style)
chord_time = 0.0
for root, chord_type in final_chords:
    chord_pitches = [root + 60, root + 64, root + 67]  # Simple triad
    for cp in chord_pitches:
        note = pretty_midi.Note(
            velocity=80, pitch=int(cp), start=chord_time, end=chord_time + 2.0
        )
        harmony_instr.notes.append(note)
    chord_time += 4.0  # Assuming 4 beats per bar

midi.instruments.append(melody_instr)
midi.instruments.append(harmony_instr)
midi.write(args.output_midi)

# --- Play Final Song ---
subprocess.run(["fluidsynth", "-ni", args.soundfont, args.output_midi, "-F", "out.wav", "-r", "44100"])
try:
    subprocess.run(["open", "out.wav"])  # macOS
except FileNotFoundError:
    subprocess.run(["aplay", "out.wav"])  # Linux alternative

