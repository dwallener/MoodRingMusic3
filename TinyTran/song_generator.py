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
from composer_harmony_trainer import TinyHarmonyTransformer, HARMONY_ROOT_VOCAB, CHORD_TYPES, DEVICE as DEVICE_HARMONY

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
parser.add_argument("--melody_key_penalty", type=float, default=5.0) # 0.5 was old default, wasn't enough
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


def build_chord_notes(root, chord_type, reference_pitch=60):
    """Build chord notes based on root, chord type, and nearby register."""
    intervals = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "dim": [0, 3, 6],
        "aug": [0, 4, 8],
        "7": [0, 4, 7, 10],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7]
    }
    chord_type = chord_type if chord_type in intervals else "maj"
    
    # Find the octave closest to the reference pitch (melody or bass line)
    base_octave = (reference_pitch // 12) * 12
    
    return [root + base_octave + interval for interval in intervals[chord_type]]


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
            if len(logits) > 1:
                for idx, interval in enumerate(INTERVAL_VOCAB):
                    test_pitch = (current_pitch + interval) % 12
                    if test_pitch not in scale_notes:
                        logits[idx] -= args.melody_key_penalty
            else:
                # Apply the penalty directly to the only value, if needed
                interval = INTERVAL_VOCAB[0]
                test_pitch = (current_pitch + interval) % 12
                if test_pitch not in scale_notes:
                    logits[0] -= args.melody_key_penalty

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
    #chords = []
    #for _ in range(bars):
    #    root = random.choice(scale_notes)
    #    chord = (root, "maj")  # Simplified to major chords for now
    #    chords.append(chord)

    harmony_input_roots = torch.tensor([[note % 12 for note in pitches[:bars * 4]]], dtype=torch.long).to(DEVICE_HARMONY)
    harmony_input_types = torch.zeros_like(harmony_input_roots).to(DEVICE_HARMONY)  # Placeholder if no prior types
    harmony_input_melody = torch.tensor([[int(p) for p in pitches[:bars * 4]]], dtype=torch.long).to(DEVICE_HARMONY)
    key_sig = torch.tensor([0]).to(DEVICE_HARMONY)  # Simplified
    mode_sig = torch.tensor([0]).to(DEVICE_HARMONY)  # 0 for Major

    with torch.no_grad():
        pred_roots, pred_types, pred_styles = harmony_model(
            harmony_input_roots, harmony_input_types, harmony_input_melody, key_sig, mode_sig
        )

    # Only take the *last step's prediction* for the harmony at this point
    predicted_root_idx = torch.argmax(pred_roots[-1]).item()
    predicted_type_idx = torch.argmax(pred_types[-1]).item()

    predicted_root = HARMONY_ROOT_VOCAB[predicted_root_idx % len(HARMONY_ROOT_VOCAB)]
    predicted_type = CHORD_TYPES[predicted_type_idx % len(CHORD_TYPES)]

    # Append directly to final chords
    final_pitches.extend(pitches)
    final_durations.extend(durations)
    final_chords.append((predicted_root, predicted_type))  # This is the correct single chord for the section

# --- MIDI Rendering ---
midi = pretty_midi.PrettyMIDI()
melody_instr = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
harmony_instrument = pretty_midi.Instrument(program=48)  # Strings Ensemble

start_time = 0.0

for pitch, duration in zip(final_pitches, final_durations):
    note = pretty_midi.Note(
        velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration
    )
    melody_instr.notes.append(note)
    start_time += duration


# Harmony Style Prediction (Whole or Broken)
current_time = 0.0 
chord_duration = 4.0
pred_styles = torch.softmax(pred_styles, dim=-1)
style_probs = pred_styles[-1].cpu().numpy()
style_choice = np.argmax(style_probs)  # 0 = Whole, 1 = Broken

root_midi_value = HARMONY_ROOT_VOCAB.index(predicted_root)  # Converts 'C' → 0, 'C#' → 1, etc.
chord_notes = build_chord_notes(root_midi_value, predicted_type, current_pitch)

if style_choice == 0:  # Whole Chord (Pad)
    for note_pitch in chord_notes:
        note = pretty_midi.Note(
            velocity=80, pitch=int(note_pitch), 
            start=current_time, end=current_time + chord_duration
        )
        harmony_instrument.notes.append(note)
else:  # Broken Chord (Arpeggio)
    # Randomly select a pattern
    arpeggio_patterns = [
        [0, 1, 2],  # Ascending
        [2, 1, 0],  # Descending
        [0, 2, 1],  # Up-Down
        [1, 0, 2],  # Random variation
    ]
    pattern = random.choice(arpeggio_patterns)
    note_time = current_time
    per_note_duration = chord_duration / len(pattern)
    
    for idx in pattern:
        note_pitch = chord_notes[idx % len(chord_notes)]
        note = pretty_midi.Note(
            velocity=80, pitch=int(note_pitch), 
            start=note_time, end=note_time + per_note_duration
        )
        harmony_instrument.notes.append(note)
        note_time += per_note_duration


midi.instruments.append(melody_instr)
midi.instruments.append(harmony_instrument)
midi.write(args.output_midi)

# --- Play Final Song ---
subprocess.run(["fluidsynth", "-ni", args.soundfont, args.output_midi, "-F", "out.wav", "-r", "44100"])
try:
    subprocess.run(["open", "out.wav"])  # macOS
except FileNotFoundError:
    subprocess.run(["aplay", "out.wav"])  # Linux alternative

