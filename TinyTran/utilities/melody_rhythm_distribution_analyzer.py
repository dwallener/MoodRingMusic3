import pandas as pd
import pretty_midi
import argparse
from collections import Counter
from tqdm import tqdm

META_CSV = "annotated_csv.csv"
MIDI_BASE = "maestro-v3.0.0/"

DURATION_BINS = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
    "thirty_second": 0.125
}

def quantize_duration(duration_in_beats):
    return min(DURATION_BINS, key=lambda x: abs(DURATION_BINS[x] - duration_in_beats))

def extract_melodic_contour(instrument):
    """Extracts the highest pitch note at each unique start time."""
    contour_notes = {}
    for note in instrument.notes:
        if note.start not in contour_notes or note.pitch > contour_notes[note.start].pitch:
            contour_notes[note.start] = note
    return list(contour_notes.values())

def analyze_rhythms(composer):
    meta = pd.read_csv(META_CSV)
    files = meta[meta['composer'].str.contains(composer, na=False)]['midi_filename'].tolist()

    raw_duration_counts = Counter()
    melodic_duration_counts = Counter()

    for fname in tqdm(files, desc=f"Analyzing {composer}"):
        try:
            midi = pretty_midi.PrettyMIDI(MIDI_BASE + fname)
            tempos = midi.get_tempo_changes()[1]
            tempo = tempos[0] if len(tempos) > 0 else 120
            seconds_per_beat = 60.0 / tempo

            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue

                # --- Raw Note Durations ---
                for note in instrument.notes:
                    duration_seconds = note.end - note.start
                    duration_beats = duration_seconds / seconds_per_beat
                    label = quantize_duration(duration_beats)
                    raw_duration_counts[label] += 1

                # --- Melodic Contour Durations ---
                melody_notes = extract_melodic_contour(instrument)
                for note in melody_notes:
                    duration_seconds = note.end - note.start
                    duration_beats = duration_seconds / seconds_per_beat
                    label = quantize_duration(duration_beats)
                    melodic_duration_counts[label] += 1

        except Exception:
            continue  # Skip problematic files

    # Display Results
    print(f"\n{'Duration':<15} | {'Raw %':>7} | {'Melodic %':>9}")
    print("-" * 36)

    total_raw = sum(raw_duration_counts.values())
    total_melodic = sum(melodic_duration_counts.values())

    for label in DURATION_BINS.keys():
        raw_count = raw_duration_counts[label]
        melodic_count = melodic_duration_counts[label]

        raw_pct = (raw_count / total_raw) * 100 if total_raw > 0 else 0
        melodic_pct = (melodic_count / total_melodic) * 100 if total_melodic > 0 else 0

        print(f"{label.title():<15} | {raw_pct:7.2f}% | {melodic_pct:9.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Rhythmic Distribution by Composer")
    parser.add_argument("--composer", type=str, required=True, help="Composer name to analyze")
    args = parser.parse_args()

    analyze_rhythms(args.composer)

