import pandas as pd
import pretty_midi
import argparse
from collections import Counter
from tqdm import tqdm

META_CSV = "annotated_csv.csv"
MIDI_BASE = "maestro-v3.0.0/"

# Duration bins in beats (assuming 4/4 time)
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

def analyze_rhythms(composer):
    meta = pd.read_csv(META_CSV)
    files = meta[meta['composer'].str.contains(composer, na=False)]['midi_filename'].tolist()

    duration_counts = Counter()

    for fname in tqdm(files, desc=f"Analyzing {composer}"):
        try:
            midi = pretty_midi.PrettyMIDI(MIDI_BASE + fname)
            
            # Estimate tempo (assuming constant tempo for simplicity)
            tempos = midi.get_tempo_changes()[1]
            if len(tempos) == 0:
                tempo = 120  # Default fallback tempo
            else:
                tempo = tempos[0]
            seconds_per_beat = 60.0 / tempo

            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    duration_seconds = note.end - note.start
                    duration_in_beats = duration_seconds / seconds_per_beat
                    label = quantize_duration(duration_in_beats)
                    duration_counts[label] += 1
        except Exception as e:
            continue  # Skip problematic files

    total = sum(duration_counts.values())
    print(f"\nRhythmic Distribution for {composer}:")
    for label in DURATION_BINS.keys():
        count = duration_counts[label]
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{label.title():<15}: {percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Rhythmic Distribution by Composer")
    parser.add_argument("--composer", type=str, required=True, help="Composer name to analyze")
    args = parser.parse_args()

    analyze_rhythms(args.composer)