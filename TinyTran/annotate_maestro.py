import pandas as pd
import os
from tqdm import tqdm
import pretty_midi
from music21 import converter, chord
import argparse

def extract_metadata(meta_csv_path, midi_base_path, output_csv_path):
    metadata_records = []
    meta_df = pd.read_csv(meta_csv_path)

    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        midi_file_path = os.path.join(midi_base_path, row['midi_filename'])
        if not os.path.exists(midi_file_path):
            continue

        record = {
            'midi_filename': row['midi_filename'],
            'composer': row['canonical_composer'],
            'duration': row['duration'],
        }

        # Extract average tempo
        try:
            midi = pretty_midi.PrettyMIDI(midi_file_path)
            tempos, _ = midi.get_tempo_changes()
            record['average_tempo'] = tempos.mean() if len(tempos) else None
        except Exception:
            record['average_tempo'] = None

        # Analyze Key, Mode, Chord Progression
        try:
            score = converter.parse(midi_file_path)
            key_analysis = score.analyze('key')
            record['key_signature'] = key_analysis.tonic.name
            record['mode'] = key_analysis.mode

            chords_seq = []
            chords = score.chordify()
            for c in chords.recurse().getElementsByClass(chord.Chord)[:16]:
                chords_seq.append(c.pitchedCommonName)
            record['chord_progression'] = chords_seq
        except Exception:
            record['key_signature'] = None
            record['mode'] = None
            record['chord_progression'] = None

        metadata_records.append(record)

    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(output_csv_path, index=False)
    print(f"Metadata saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate MAESTRO MIDI dataset with metadata.")
    parser.add_argument("--meta_csv", required=True, help="Path to maestro-v3.0.0.csv file.")
    parser.add_argument("--midi_base", required=True, help="Base path to the unzipped MAESTRO MIDI files.")
    parser.add_argument("--output_csv", required=True, help="Path to save the annotated metadata CSV.")

    args = parser.parse_args()

    extract_metadata(args.meta_csv, args.midi_base, args.output_csv)