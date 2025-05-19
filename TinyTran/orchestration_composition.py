import torch
import torch.nn as nn
import json
import random
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Setting up for mood-based key transposition
KEY_MAP = {
    "Focus": ["C", "A", "F"],
    "Relax": ["F", "Eb", "Bb"],
    "Energy": ["G", "D", "E"],
    "Sleep": ["A Minor", "C Minor", "G Minor"],
    "Creative Flow": ["E", "G", "D"],
    "Calm Confidence": ["C", "F", "Bb"],
    "Romantic": ["Bb", "D minor", "A Minor"],
    "Reflective": ["C Minor", "D Minor", "Eb", "G Minor"]
}

key_offsets = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

def select_starting_key(mood):
    possible_keys = KEY_MAP.get(mood, ["C"])  # Default to C if mood not found
    return random.choice(possible_keys)

def calculate_transposition_offset(starting_key):
    key_offsets = {
        "C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
        "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11
    }
    return key_offsets.get(starting_key, 0)


# -----------------------
# Tiny Transformer Model
# -----------------------
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
        return self.fc_out(out)

# -----------------------
# Utility Functions
# -----------------------
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
        probs = torch.softmax(logits[0, -1] / temperature, dim=0).detach().numpy()
        next_token = random.choices(range(len(probs)), weights=probs, k=1)[0]
        if next_token == 0:
            break
        sequence.append(next_token)
    return [idx_to_token[t] for t in sequence if t != 0]

def load_transition_matrix(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def choose_next_phrase(current_state, matrix):
    transitions = matrix.get(current_state, {})
    if not transitions:
        return "Silent"
    choices, weights = zip(*transitions.items())
    return random.choices(choices, weights=weights, k=1)[0]

def transpose_pitch(pitch, target_key, mode):
    key_offsets = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                   "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    offset = key_offsets.get(target_key.upper(), 0)
    return pitch + offset

def register_shift(pitch, register, instrument):
    base_shifts = {"Cello": -24, "Viola": -12, "Violin1": 0, "Violin2": 0}
    shift = base_shifts.get(instrument, 0)
    if register == "low":
        shift -= 12
    elif register == "high":
        shift += 12
    return max(0, min(127, pitch + shift))

instrument_ranges = {
    "Cello": (36, 60),    # C2 to C4
    "Viola": (48, 72),    # C3 to C5
    "Violin1": (55, 84),  # G3 to C6
    "Violin2": (55, 84),
}

def clamp_pitch_octave_up(pitch, instrument):
    min_pitch, max_pitch = instrument_ranges.get(instrument, (0, 127))
    while pitch < min_pitch:
        pitch += 12
    while pitch > max_pitch:
        pitch -= 12
    return max(min_pitch, min(max_pitch, pitch))

def get_allowed_pitches(key, mode):
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]
    base_offset = key_offsets.get(key.upper(), 0)
    scale = major_scale if mode.lower() == "major" else minor_scale
    return [(note + base_offset) % 12 for note in scale]

def infer_chord(bass_notes):
    return max(set(bass_notes), key=bass_notes.count) if bass_notes else 60  # Default C if none

def display_composition_settings(mood, starting_key, resolved_sections):
    print("\n🎼 Composition Settings:")
    print(f"- Mood: {mood}")
    print(f"- Starting Key: {starting_key}\n")
    print("📖 Section Key Resolutions:")
    for section_name, key in resolved_sections.items():
        print(f"- {section_name}: {key}")
    print("\n" + "-" * 40 + "\n")


# -----------------------
# Orchestration Logic
# -----------------------
def orchestrate(output_file="mozart_composition_mido.mid",
                transition_matrix_file="cello_transition_matrix.json",
                song_structure_file="song_structure.json",
                mood="Romantic"):

    starting_key = select_starting_key(mood)
    transposition_offset = calculate_transposition_offset(starting_key)
    print(f"🎵 Selected Mood: {mood}")
    print(f"🎶 Starting Key: {starting_key} (Offset: {transposition_offset} semitones)")


    instruments = {
        "Cello": 32,
        "Viola": 41,
        "Violin1": 40,
        "Violin2": 40
    }
    phrase_lengths = {"Short": 8, "Medium": 16, "Long": 32}
    ticks_per_beat = 480
    matrix = load_transition_matrix(transition_matrix_file)

    with open(song_structure_file, "r") as f:
        sections = json.load(f)

    tempo_bpm = 120
    if "tempo" in sections[0]:
        tempo_bpm = sections[0]["tempo"]
    tempo = mido.bpm2tempo(tempo_bpm)

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    bass_notes_by_measure = []
    current_measure_notes = []
    ticks_accumulated = 0

    for track_name, program in instruments.items():
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        track.append(Message('program_change', program=program, time=0))

        for section in sections:
            bars_remaining = section["length_bars"]
            temperature = section["temperature"]
            register = section["register"]
            key = section["key"]

            # shift to mood-based key
            resolved_key_offset = calculate_transposition_offset(section["key"]) + transposition_offset
            resolved_key_offset = resolved_key_offset % 12  # Wrap around
            resolved_key_name = [k for k, v in key_offsets.items() if v == resolved_key_offset][0]

            print(f"→ Section: {section['section']}, Key: {resolved_key_name}, Mode: {section['mode']}, Tempo Factor: {section['temperature']}")

            mode = section["mode"]
            allowed_pitches = get_allowed_pitches(key, mode)

            current_state = "Short"
            time_cursor = 0

            print(f"\n--- Generating Section: {section.get('section', 'Unnamed')} ---")
            print(f"Key: {key} {mode.capitalize()} | Tempo: {tempo_bpm} BPM | Register: {register} | Temperature: {temperature}")
            print(f"Bars: {section['length_bars']}")

            while bars_remaining > 0:
                phrase_type = choose_next_phrase(current_state, matrix)
                current_state = phrase_type

                if phrase_type == "Silent":
                    rest_ticks = 4 * ticks_per_beat
                    track.append(Message('note_off', time=rest_ticks))
                    bars_remaining -= 1
                    continue

                model_prefix = f"{track_name.lower()}_{phrase_type.lower()}"
                model_file = f"{model_prefix}.model.pt"
                vocab_file = f"{model_prefix}.vocab.json"

                model, token_to_idx, idx_to_token = load_model_and_vocab(model_file, vocab_file)
                tokens = generate_phrase(model, token_to_idx, idx_to_token,
                                         max_length=phrase_lengths[phrase_type],
                                         temperature=temperature)

                total_phrase_quarters = 0
                for token in tokens:
                    pitch_midi, dur_sixteenths = map(int, token.split("_"))
                    #pitch_midi = transpose_pitch(pitch_midi, key, mode)
                    pitch_midi = transpose_pitch(pitch_midi, resolved_key_name, mode)
                    pitch_class = pitch_midi % 12

                    # Enforce staying in key
                    if pitch_class not in allowed_pitches:
                        if dur_sixteenths <= 2:
                            continue  # Skip short wrong notes
                        if random.random() > 0.1:
                            continue  # Allow rare wrong notes

                    pitch_midi = register_shift(pitch_midi, register, track_name)
                    pitch_midi = clamp_pitch_octave_up(pitch_midi, track_name)
                    duration = dur_sixteenths * int(ticks_per_beat / 4)

                    track.append(Message('note_on', note=pitch_midi, velocity=64, time=time_cursor))
                    track.append(Message('note_off', note=pitch_midi, velocity=64, time=duration))

                    time_cursor = 0
                    total_phrase_quarters += dur_sixteenths / 4

                    if track_name == "Cello":
                        current_measure_notes.append(pitch_midi)
                        ticks_accumulated += duration
                        if ticks_accumulated >= 4 * ticks_per_beat:
                            bass_notes_by_measure.append(current_measure_notes)
                            current_measure_notes = []
                            ticks_accumulated = 0

                bars_remaining -= total_phrase_quarters / 4
                if bars_remaining < 0:
                    overshoot_ticks = int(abs(bars_remaining) * 4 * ticks_per_beat)
                    track.append(Message('note_off', time=overshoot_ticks))
                    bars_remaining = 0

    if current_measure_notes:
        bass_notes_by_measure.append(current_measure_notes)

    # Add Chord Marker Track
    chord_track = MidiTrack()
    mid.tracks.append(chord_track)
    chord_track.append(MetaMessage('track_name', name="Chord Markers", time=0))
    chord_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    chord_track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))

    marker_interval = 4 * ticks_per_beat
    for measure_notes in bass_notes_by_measure:
        root_midi = infer_chord(measure_notes)
        chord_track.append(Message('note_on', note=root_midi, velocity=1, time=0))
        chord_track.append(Message('note_off', note=root_midi, velocity=1, time=marker_interval))

    mid.save(output_file)
    print(f"MIDI composition exported to {output_file}")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    orchestrate()

