import os
import sys
import json

REST_THRESHOLD = 0.5
HOLD_THRESHOLD = 1.0

def classify_phrase_length(length):
    if length <= 8:
        return "Short"
    elif length <= 24:
        return "Medium"
    else:
        return "Long"

def extract_phrases_from_spine(spine_lines):
    phrases = []
    current_phrase = []

    for token in spine_lines:
        if token == "rest" or token == ".":
            if current_phrase:
                phrases.append(current_phrase)
                current_phrase = []
        else:
            current_phrase.append(token)

    if current_phrase:
        phrases.append(current_phrase)

    return phrases

def process_kern_file(file_path, output_folder):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("**")]

    # Transpose columns (spines)
    spines = list(zip(*[line.split('\t') for line in lines if not line.startswith('=')]))

    for spine_index, spine_data in enumerate(spines):
        phrases = extract_phrases_from_spine(spine_data)

        # Classify phrases into short, medium, long
        phrase_classes = {"Short": [], "Medium": [], "Long": []}
        for phrase in phrases:
            phrase_len = sum(int(token.split("_")[1]) for token in phrase if "_" in token)
            phrase_type = classify_phrase_length(phrase_len)
            phrase_classes[phrase_type].append(phrase)

        for cls, cls_phrases in phrase_classes.items():
            if cls_phrases:
                output_filename = f"Spine{spine_index + 1}_{cls}.json"
                output_path = os.path.join(output_folder, output_filename)
                with open(output_path, "w") as out_f:
                    json.dump(cls_phrases, out_f)
                print(f"[Saved] {output_filename} with {len(cls_phrases)} phrases.")
            else:
                print(f"[Info] No {cls} phrases for Spine{spine_index + 1}. Skipping JSON export.")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(".krn")]
    if not files:
        print("[Warning] No .krn files found in the input folder.")
        return

    for file in files:
        print(f"[Info] Processing {file}")
        process_kern_file(os.path.join(input_folder, file), output_folder)

    print(f"[Success] Processed {len(files)} files.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prep_phrase_datasets.py <input_folder> <output_folder>")
    else:
        process_folder(sys.argv[1], sys.argv[2])