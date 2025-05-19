import music21
import os
import sys

def midi_to_pseudo_kern(midi_file_path, output_path=None):
    midi = music21.converter.parse(midi_file_path)
    parts = midi.parts.stream()
    spines = [f"**Spine{i+1}" for i in range(len(parts))]
    
    # Prepare per-part events list and beat tracking
    events = [[] for _ in parts]
    beat_counters = [0 for _ in parts]
    barlines = []
    
    for idx, part in enumerate(parts):
        notes = list(part.flat.notesAndRests)
        for n in notes:
            dur_quarters = n.quarterLength
            beat_counters[idx] += dur_quarters

            if isinstance(n, music21.note.Note):
                token = f"{n.pitch.midi}_{int(dur_quarters * 4)}"
            elif isinstance(n, music21.note.Rest):
                token = "rest"
            else:
                continue

            events[idx].append(token)

            # Insert barline if 4 beats accumulated
            if beat_counters[idx] >= 4:
                events[idx].append("=")
                beat_counters[idx] -= 4

        # Ensure ending barline
        if not events[idx] or events[idx][-1] != "=":
            events[idx].append("=")

    # Determine maximum number of events per spine to pad properly
    max_length = max(len(e) for e in events)
    for e in events:
        e.extend(["."] * (max_length - len(e)))

    # Prepare output lines
    lines = []
    lines.append("\t".join(spines))

    for row in range(max_length):
        lines.append("\t".join(events[col][row] for col in range(len(spines))))

    final_output = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(final_output)
        print(f"Saved to {output_path}")
    else:
        print(final_output)

def process_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".mid") or file.endswith(".midi"):
            midi_path = os.path.join(folder_path, file)
            output_path = os.path.splitext(midi_path)[0] + ".krn"
            midi_to_pseudo_kern(midi_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python midi_to_pseudo_kern.py <midi_folder_or_file>")
    else:
        target = sys.argv[1]
        if os.path.isdir(target):
            process_folder(target)
        else:
            midi_to_pseudo_kern(target)