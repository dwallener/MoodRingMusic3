import torch
import pretty_midi
import random
from composer_melody_trainer import (
    TinyMelodyTransformer, INTERVAL_VOCAB, DURATION_VOCAB, DEVICE
)

CHECKPOINT_PATH = "checkpoints/Mozart_epoch500.pt"
OUTPUT_MIDI = "stylized_melody.mid"
SOUNDFONT_PATH = "../soundfonts/FluidR3_GM.sf2"

# Load Model
model = TinyMelodyTransformer().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --- Step 1: Create a Raw, Random Sequence (Replace this with any input) ---
raw_seed_intervals = [random.choice(INTERVAL_VOCAB) for _ in range(32)]
durations = [random.choice(DURATION_VOCAB) for _ in range(32)]

input_seq = torch.tensor(
    [INTERVAL_VOCAB.index(i) for i in raw_seed_intervals], dtype=torch.long
).unsqueeze(1).to(DEVICE)

# --- Step 2: Stylize the Sequence ---
stylized_intervals = []
stylized_durations = []
current_pitch = 60  # Middle C
pitches = []

with torch.no_grad():
    pred_intervals, pred_durations, pred_registers = model(input_seq)
    for t in range(32):
        next_token_logits = pred_intervals[t]
        next_token = torch.argmax(next_token_logits).item()
        next_interval = INTERVAL_VOCAB[next_token] if next_token < len(INTERVAL_VOCAB) else 0
        stylized_intervals.append(next_interval)

        next_duration_logits = pred_durations[t]
        next_duration_token = torch.argmax(next_duration_logits).item()
        next_duration = DURATION_VOCAB[next_duration_token]
        stylized_durations.append(next_duration)

        current_pitch += next_interval
        current_pitch = max(0, min(127, current_pitch))
        pitches.append(current_pitch)

# --- Step 3: Create MIDI Output ---
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
start_time = 0.0

for pitch, duration in zip(pitches, stylized_durations):
    note = pretty_midi.Note(
        velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration
    )
    instrument.notes.append(note)
    start_time += duration

midi.instruments.append(instrument)
midi.write(OUTPUT_MIDI)

# Optional: Play the result immediately
import subprocess
subprocess.run(["fluidsynth", "-ni", SOUNDFONT_PATH, OUTPUT_MIDI, "-F", "out.wav", "-r", "44100"])
subprocess.run(["open", "out.wav"])  # For MacOS

