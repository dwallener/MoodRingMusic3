import music21
import os
import sys
import json

# Thresholds for phrase boundaries
REST_THRESHOLD = 0.5
HOLD_THRESHOLD = 1.0

PART_INDEX = {
    "cello": 0,
    "viola": 1,
    "violin1": 2,
    "violin2": 3
}

def classify_phrase_length(length):
    if length <= 8:
        return "Short"
    elif length <= 24:
        return "Medium"
    else:
        return "Long"

def extract_phrases(file_path, instrument_role):
    score = music21.converter.parse(file_path)

    try:
        target_index = PART_INDEX[instrument_role.lower()]
        part = score.parts[target_index]
    except (KeyError, IndexError):
        print(f"[Warning] Instrument role '{instrument_role}' not found or invalid in file {file_path}. Skipping.")
        return []

    phrases = []
    current_phrase = []

    for n in part.flat.notesAndRests:
        if isinstance(n, music21.note.Rest):
            if n.quarterLength >= REST_THRESHOLD * get_measure_length(n):
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []
        elif isinstance(n, music21.note.Note):
            dur = int(n.quarterLength * 4)  # Normalize to sixteenth notes
            token = f"{n.pitch.midi}_{dur}"
            current_phrase.append(token)

    if current_phrase:
        phrases.append(current_phrase)

    return phrases

def get_measure_length(element):
    measure = element.getContextByClass('Measure')
    if measure and measure.timeSignature:
        return measure.timeSignature.barDuration.quarterLength
    return 4.0  # Assume 4/4 if unknown

def process_folder(folder_path, instrument_role):
    phrases_by_length = {"Short": [], "Medium": [], "Long": []}

    for file in os.listdir(folder_path):
        if file.endswith(".krn"):
            phrases = extract_phrases(os.path.join(folder_path, file), instrument_role)
            for phrase in phrases:
                length = sum(int(token.split('_')[1]) for token in phrase)
                category = classify_phrase_length(length)
                phrases_by_length[category].append(phrase)

    for category in ["Short", "Medium", "Long"]:
        output_file = f"{instrument_role.lower()}_phrase_dataset_{category.lower()}.json"
        with open(output_file, "w") as f:
            json.dump(phrases_by_length[category], f)
        print(f"Saved {len(phrases_by_length[category])} {category} phrases to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_phrase_dataset.py <krn_folder> <instrument_role (cello|viola|violin1|violin2)>")
    else:
        process_folder(sys.argv[1], sys.argv[2])