import music21
import os
import sys
import json

# Thresholds
REST_THRESHOLD = 0.5
HOLD_THRESHOLD = 1.0
SILENCE_THRESHOLD = 1.0

def classify_phrase_length(length):
    if length <= 8:
        return "Short"
    elif length <= 24:
        return "Medium"
    else:
        return "Long"

def extract_phrases(file_path):
    score = music21.converter.parse(file_path)
    violin_parts = [p for p in score.parts if any("Violin" in str(inst) for inst in p.getInstruments())]

    if not violin_parts or len(violin_parts) < 2:
        return []

    violin2 = violin_parts[1]
    phrases = []
    current_phrase = []

    for n in violin2.flat.notesAndRests:
        if isinstance(n, music21.note.Rest):
            rest_length = n.quarterLength
            if rest_length >= REST_THRESHOLD * get_measure_length(n):
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []
            else:
                continue
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
    return 4.0  # Default to 4/4

def process_folder(folder_path, output_json="phrase_dataset.json"):
    all_phrases = []
    for file in os.listdir(folder_path):
        if file.endswith(".krn"):
            phrases = extract_phrases(os.path.join(folder_path, file))
            all_phrases.extend(phrases)

    with open(output_json, "w") as f:
        json.dump(all_phrases, f)

    print(f"Extracted {len(all_phrases)} phrases to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_phrases_for_training.py <krn_folder>")
    else:
        process_folder(sys.argv[1])

