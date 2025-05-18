import torch
import random
from composer_melody_trainer import (
    TinyMelodyTransformer, INTERVAL_VOCAB, DURATION_VOCAB, DEVICE, build_dataset
)

CHECKPOINT_PATH = "checkpoints/Mozart_epoch105.pt"  # Update as needed
SEED_LENGTH = 4
GENERATE_LENGTH = 32

# Load Model
model = TinyMelodyTransformer().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load Validation Data for Seed
sequences = build_dataset()
split = int(0.8 * len(sequences))
val_data = sequences[split:]
seed, *_ = random.choice(val_data)

print(f"\n🎵 Seed Intervals: {seed}")

# Prepare Seed Input
generated_intervals = seed[:]
input_seq = torch.tensor(
    [INTERVAL_VOCAB.index(i) for i in seed], dtype=torch.long
).unsqueeze(1).to(DEVICE)

current_pitch = 60  # Start from Middle C
pitches = [current_pitch]

# Generate Sequence
for _ in range(GENERATE_LENGTH):
    with torch.no_grad():
        pred_intervals, pred_durations, pred_registers = model(input_seq)
        next_token_logits = pred_intervals[-1]
        next_token = torch.argmax(next_token_logits).item()
        next_interval = INTERVAL_VOCAB[next_token] if next_token < len(INTERVAL_VOCAB) else 0
        generated_intervals.append(next_interval)

        input_seq = torch.cat(
            [input_seq, torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)], dim=0
        )

        # Update current pitch and store
        current_pitch += next_interval
        current_pitch = max(0, min(127, current_pitch))
        pitches.append(current_pitch)

# Print Results
print(f"\n🎶 Generated Intervals: {generated_intervals}")
print(f"\n🎼 Absolute Pitches: {pitches}")

# Optional: Calculate Durations and Registers
durations = []
registers = []
for idx in range(len(generated_intervals)):
    with torch.no_grad():
        pred_intervals, pred_durations, pred_registers = model(input_seq[:idx+1])
        # Duration
        dur_token = torch.argmax(pred_durations[-1]).item()
        durations.append(DURATION_VOCAB[dur_token])
        # Register
        reg_token = torch.argmax(pred_registers[-1]).item()
        reg_label = ["low", "mid", "high"][reg_token]
        registers.append(reg_label)

print(f"\n⏱️ Durations: {durations}")
print(f"\n📚 Registers: {registers}")
