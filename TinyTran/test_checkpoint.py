import torch
import pretty_midi
import subprocess
import random
from composer_melody_trainer import (
    TinyMelodyTransformer, 
    INTERVAL_VOCAB, 
    DURATION_VOCAB, 
    DEVICE, 
    build_dataset
)

CHECKPOINT_PATH = "checkpoints/Mozart_epoch500.pt"  # Update as needed
OUTPUT_MIDI = "generated_melody.mid"
SOUNDFONT_PATH = "../soundfonts/FluidR3_GM.sf2"    # Update this path to your SoundFont

# Load Model
model = TinyMelodyTransformer().to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

#model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# Get Validation Data
sequences = build_dataset()
split = int(0.8 * len(sequences))
val_data = sequences[split:]
seed, *_ = random.choice(val_data)

# Generate Continuation
generated = seed[:]
durations = [0.5] * len(seed)  # Default duration for seed notes
input_seq = torch.tensor([INTERVAL_VOCAB.index(i) for i in seed], dtype=torch.long).unsqueeze(1).to(DEVICE)

current_pitch = 60  # Starting at Middle C
pitches = []

for _ in range(32):
    with torch.no_grad():
        pred_intervals, pred_durations, pred_registers = model(input_seq)

        # Interval Prediction
        temperature = 0.8
        next_token_logits = pred_intervals[-1]
        next_token = torch.argmax(next_token_logits).item()
        next_interval = INTERVAL_VOCAB[next_token] if next_token < len(INTERVAL_VOCAB) else 0
        generated.append(next_interval)

        # Duration Prediction
        next_duration_logits = pred_durations[-1]
        next_duration_token = torch.argmax(next_duration_logits).item()
        next_duration = DURATION_VOCAB[next_duration_token]
        durations.append(next_duration)

        # Register Prediction (currently not applied in this generation loop, but could be used for constraints)

        input_seq = torch.cat([input_seq, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=0)

        # Update pitch
        current_pitch += next_interval
        current_pitch = max(0, min(127, current_pitch))  # Clamp to valid MIDI pitch range
        pitches.append(current_pitch)

# Create PrettyMIDI Object
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
start_time = 0.0

for pitch, duration in zip(pitches, durations):
    note = pretty_midi.Note(
        velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration
    )
    instrument.notes.append(note)
    start_time += duration

midi.instruments.append(instrument)
midi.write(OUTPUT_MIDI)

# Play with Fluidsynth and system audio player
subprocess.run(["fluidsynth", "-ni", SOUNDFONT_PATH, OUTPUT_MIDI, "-F", "out.wav", "-r", "44100"])

# MacOS:
subprocess.run(["open", "out.wav"])
# Linux:
# subprocess.run(["aplay", "out.wav"])
# MacOS alternative:
# subprocess.run(["afplay", "out.wav"])