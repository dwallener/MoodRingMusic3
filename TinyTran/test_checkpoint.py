import random
import torch
import pretty_midi
import subprocess
from composer_melody_trainer import TinyMelodyTransformer, INTERVAL_VOCAB, DEVICE, build_dataset

CHECKPOINT_PATH = "checkpoints/Mozart_epoch100.pt"  # Update as needed
OUTPUT_MIDI = "generated_melody.mid"
SOUNDFONT_PATH = "/path/to/FluidR3_GM.sf2"  # Update with correct SoundFont path

# Load Model
model = TinyMelodyTransformer().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# Get Validation Data
sequences = build_dataset()
split = int(0.8 * len(sequences))
val_data = sequences[split:]
seed, _ = random.choice(val_data)

# Generate Continuation
generated = seed[:]
input_seq = torch.tensor([INTERVAL_VOCAB.index(i) for i in seed], dtype=torch.long).unsqueeze(1).to(DEVICE)

for _ in range(32):
    with torch.no_grad():
        output = model(input_seq).squeeze(1)
        next_token_logits = output[-1]
        next_token = torch.argmax(next_token_logits).item()
        next_interval = INTERVAL_VOCAB[next_token] if next_token < len(INTERVAL_VOCAB) else 0
        generated.append(next_interval)
        input_seq = torch.cat([input_seq, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=0)

# Convert to Absolute MIDI Pitches (starting from middle C, 60)
pitches = []
current_pitch = 60  # Starting pitch (Middle C)
for interval in generated:
    current_pitch += interval
    pitches.append(current_pitch)

# Create PrettyMIDI Object
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
start_time = 0.0
duration = 0.5  # Fixed duration per note for simplicity

for pitch in pitches:
    note = pretty_midi.Note(
        velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration
    )
    instrument.notes.append(note)
    start_time += duration

midi.instruments.append(instrument)
midi.write(OUTPUT_MIDI)

# Play with Fluidsynth
subprocess.run(["fluidsynth", "-ni", SOUNDFONT_PATH, OUTPUT_MIDI, "-F", "out.wav", "-r", "44100"])
subprocess.run(["aplay", "out.wav"])  # Or use any system audio player you prefer

