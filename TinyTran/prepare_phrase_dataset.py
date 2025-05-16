import music21
import os
import pickle

# Constants
INTERVAL_RANGE = range(-12, 13)  # -12 to +12 semitones
DURATION_BUCKETS = [0.25, 0.5, 1.0, 2.0, 4.0]  # Sixteenth to Whole notes
REST_THRESHOLD = 0.5  # Half measure rest = phrase break
HOLD_THRESHOLD = 1.0  # One measure hold = phrase break

def tokenize(interval, duration):
    if interval not in INTERVAL_RANGE:
        return None  # Skip out-of-range intervals
    try:
        duration_index = DURATION_BUCKETS.index(duration)
    except ValueError:
        return None  # Skip unknown durations
    return (interval + 12) * len(DURATION_BUCKETS) + duration_index

def extract_phrases(file_path):
    score = music21.converter.parse(file_path)
    violin_parts = [p for p in score.parts if any("Violin" in str(inst) for inst in p.getInstruments())]

    if len(violin_parts) < 2:
        return []

    violin2 = violin_parts[1]
    phrases = []
    current_phrase = []
    last_pitch = None

    for n in violin2.flat.notesAndRests:
        measure_length = get_measure_length(n)

        if isinstance(n, music21.note.Rest):
            if n.quarterLength >= REST_THRESHOLD * measure_length:
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []
            continue  # Skip rests in phrase content

        if isinstance(n, music21.note.Note):
            interval = 0 if last_pitch is None else n.pitch.midi - last_pitch
            duration = min(DURATION_BUCKETS, key=lambda x: abs(x - n.quarterLength))  # Snap to closest duration
            token = tokenize(interval, duration)

            if token is not None:
                current_phrase.append(token)

            last_pitch = n.pitch.midi

            if n.quarterLength >= HOLD_THRESHOLD * measure_length:
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []

    if current_phrase:
        phrases.append(current_phrase)

    return phrases

def get_measure_length(element):
    measure = element.getContextByClass('Measure')
    if measure and measure.timeSignature:
        return measure.timeSignature.barDuration.quarterLength
    return 4.0  # Default to 4/4 time

def prepare_dataset(input_dir, output_file):
    all_phrases = []
    for file in os.listdir(input_dir):
        if file.endswith(".krn"):
            file_path = os.path.join(input_dir, file)
            phrases = extract_phrases(file_path)
            all_phrases.extend(phrases)

    with open(output_file, "wb") as f:
        pickle.dump(all_phrases, f)

    print(f"Saved {len(all_phrases)} phrases to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python prepare_phrase_dataset.py <krn_folder> <output_pickle_file>")
    else:
        prepare_dataset(sys.argv[1], sys.argv[2])

