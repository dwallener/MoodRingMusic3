import mido
from mido import Message, MidiFile, MidiTrack
import random

# 🎵 Instrument Assignments by Genre (General MIDI Program Numbers)
GENRE_INSTRUMENTS = {
    "Classical": {"melody": 40, "chords_primary": 41, "chords_secondary": 42, "bass": 43},
    "Jazz": {"melody": 56, "chords_primary": 0, "chords_secondary": 24, "bass": 33},
    "Pop": {"melody": 4, "chords_primary": 27, "chords_secondary": 81, "bass": 34},
    "Dance": {"melody": 80, "chords_primary": 81, "chords_secondary": 89, "bass": 38}
}

SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11], 'minor': [0, 2, 3, 5, 7, 8, 10], 'dorian': [0, 2, 3, 5, 7, 9, 10],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10], 'lydian': [0, 2, 4, 6, 7, 9, 11], 'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10], 'harmonic minor': [0, 2, 3, 5, 7, 8, 11], 'pentatonic': [0, 2, 4, 7, 9]
}

SCALE_MAP = { 
    "Focus": ["dorian", "mixolydian", "major"], "Relax": ["lydian", "major", "pentatonic"], 
    "Energy": ["major", "mixolydian", "phrygian"], "Sleep": ["aeolian", "dorian", "pentatonic"], 
    "Creative Flow": ["lydian", "mixolydian", "dorian"], "Calm Confidence": ["major", "dorian"], 
    "Romantic": ["harmonic minor", "aeolian", "major"], "Reflective": ["aeolian", "dorian", "pentatonic"] 
}

KEY_MAP = { 
    "Focus": ["C", "A", "F"], "Relax": ["F", "Eb", "Bb"], "Energy": ["G", "D", "E"], 
    "Sleep": ["A", "C", "G"], "Creative Flow": ["E", "G", "D"], "Calm Confidence": ["C", "F", "Bb"], 
    "Romantic": ["Bb", "D", "A"], "Reflective": ["C", "D", "Eb", "G"] 
}

CHORD_PROGRESSIONS = {
    "Focus": ["I-vi-IV-V", "ii-V-I", "Modal Dorian"], "Relax": ["I-IV-I", "I-vi-IV-I", "Lydian Static"],
    "Energy": ["I-V-vi-IV", "IV-V-I", "Dominant Cycle"], "Sleep": ["I-ii-I", "Pedal Tones", "Open 5ths"],
    "Creative Flow": ["I-V-vi-IV", "ii-V-I", "Modal Mixolydian"], "Calm Confidence": ["I-IV-V-I", "ii-V-I", "I-vi-ii-V"],
    "Romantic": ["I-vi-IV-V", "ii-V-I", "I-IV-I"], "Reflective": ["I-vi-IV-I", "Modal Aeolian", "I-ii-IV-V"]
}

SONG_STRUCTURES = {
    "Pop": ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus", "Outro"],
    "Dance": ["Intro", "Build", "Drop", "Break", "Drop", "Outro"],
    "Classical": ["Intro", "Theme", "Development", "Recapitulation", "Coda"],
    "Jazz": ["Head", "Solo", "Solo", "Head", "Outro"]
}


class UniversalMelodyGenerator:
    def __init__(self, base_note=60):
        self.base_note = base_note
        self.motif = None  # Store motif here

    def generate_full_song(self, goal='Focus', genre='Classical'):
        structure = SONG_STRUCTURES.get(genre, SONG_STRUCTURES["Pop"])
        full_midi = MidiFile()
        tracks = [MidiTrack() for _ in range(4)]
        for t in tracks:
            full_midi.tracks.append(t)

        instrument_set = GENRE_INSTRUMENTS.get(genre, GENRE_INSTRUMENTS["Classical"])
        for idx, (track_name, channel) in enumerate(zip(["melody", "chords_primary", "chords_secondary", "bass"], range(4))):
            tracks[idx].append(Message('program_change', program=instrument_set[track_name], time=0, channel=channel))

        total_sections = len(structure)
        for idx, section in enumerate(structure):
            mid, tempo, key, mode, progression = self.generate_section(
                goal, genre, section, section_index=idx, total_sections=total_sections
            )
            for i, track in enumerate(mid.tracks):
                for msg in track:
                    tracks[i].append(msg)

        return full_midi, structure

    def generate_section(self, goal, genre, section, section_index=0, total_sections=1):
        energy_factor = section_index / max(1, total_sections - 1)
        base_tempo = random.randint(60, 140)
        energy_tempo_boost = int(energy_factor * 30)
        tempo_mod = {"Intro": -10, "Verse": 0, "Chorus": +10, "Bridge": +5, "Outro": -5}.get(section, 0)
        tempo = max(60, base_tempo + tempo_mod + energy_tempo_boost)

        base_length = 8
        note_count = int(base_length + energy_factor * 8)

        scale_choice = random.choice(SCALE_MAP[goal])
        key_choice = random.choice(KEY_MAP[goal])
        progression_choice = random.choice(CHORD_PROGRESSIONS[goal])

        mode = scale_choice.lower()
        key_offset = self.note_to_midi_offset(key_choice)
        scale = SCALES[mode]

        mid = MidiFile()
        tracks = [MidiTrack() for _ in range(4)]
        for t in tracks:
            mid.tracks.append(t)

        tempo_microsec = mido.bpm2tempo(tempo)
        for t in tracks:
            t.append(mido.MetaMessage('set_tempo', tempo=tempo_microsec))

        # Generate or reuse motif
        if section in ["Intro", "Verse"] and self.motif is None:
            self.motif = self.create_motif(scale)
            motif_sequence = self.motif
        elif section in ["Chorus", "Bridge", "Outro"] and self.motif:
            motif_sequence = self.transform_motif(self.motif, section)
        else:
            motif_sequence = self.random_contour(length=note_count)

        rhythm = self.random_rhythm(length=len(motif_sequence))
        current_note = self.base_note + key_offset

        for interval, duration in zip(motif_sequence, rhythm):
            current_note += interval
            scale_note = self.snap_to_scale(current_note, scale)
            ticks = self.beats_to_ticks(duration, tempo, mid.ticks_per_beat)
            velocity = int(60 + energy_factor * 40)
            tracks[0].append(Message('note_on', note=scale_note, velocity=velocity, time=0, channel=0))
            tracks[0].append(Message('note_off', note=scale_note, velocity=velocity, time=ticks, channel=0))

        self.add_chords(tracks[1:], progression_choice, key_offset, tempo, mid.ticks_per_beat, sum(rhythm))
        return mid, tempo, key_choice, mode, progression_choice

    def create_motif(self, scale, length=4):
        """Create a short, recognizable motif (melodic contour)."""
        return self.random_contour(length=length)

    def transform_motif(self, motif, section):
        """Develop motif based on section type."""
        new_motif = motif.copy()
        if section == "Chorus":
            return [n + random.choice([-2, 2, 4]) for n in new_motif]  # Transpose
        elif section == "Bridge":
            return [-n for n in reversed(new_motif)]  # Invert and reverse
        elif section == "Outro":
            return [n // 2 for n in new_motif]  # Compress intervals
        return new_motif

    def random_contour(self, length=8, max_interval=5):
        return [random.randint(-max_interval, max_interval) for _ in range(length)]

    def random_rhythm(self, length=8):
        return [random.choice([0.5, 1, 1.5, 2]) for _ in range(length)]

    def add_chords(self, tracks, progression, key_offset, tempo, ticks_per_beat, total_duration):
        chords = self.resolve_progression(progression)
        chord_duration = total_duration / len(chords)
        chord_ticks = self.beats_to_ticks(chord_duration, tempo, ticks_per_beat)

        for chord in chords:
            chord_notes = self.get_chord_notes(chord, key_offset)
            bass_note = chord_notes[0] - 24

            for note in chord_notes:
                tracks[0].append(Message('note_on', note=note, velocity=70, time=0, channel=1))
            for note in [n + 12 for n in chord_notes]:
                tracks[1].append(Message('note_on', note=note, velocity=50, time=0, channel=2))
            tracks[2].append(Message('note_on', note=bass_note, velocity=80, time=0, channel=3))

            for note in chord_notes:
                tracks[0].append(Message('note_off', note=note, velocity=70, time=chord_ticks, channel=1))
            for note in [n + 12 for n in chord_notes]:
                tracks[1].append(Message('note_off', note=note, velocity=50, time=chord_ticks, channel=2))
            tracks[2].append(Message('note_off', note=bass_note, velocity=80, time=chord_ticks, channel=3))

    def resolve_progression(self, progression):
        custom_progressions = {
            "Modal Dorian": ["i", "IV", "v", "i"], "Modal Mixolydian": ["I", "bVII", "IV", "I"],
            "Lydian Static": ["I", "II", "I", "II"], "Dominant Cycle": ["V", "I", "V", "I"],
            "Pedal Tones": ["I", "I", "I", "I"], "Open 5ths": ["I5", "V5", "IV5", "I5"],
            "Modal Aeolian": ["i", "VI", "VII", "i"]
        }
        return custom_progressions.get(progression, progression.split('-'))

    def get_chord_notes(self, chord_symbol, key_offset):
        CHORD_MAP = {'I': 0, 'ii': 2, 'iii': 4, 'IV': 5, 'V': 7, 'vi': 9, 'vii°': 11}
        root = CHORD_MAP.get(chord_symbol.strip('i').upper(), 0)
        if chord_symbol.endswith('5'):
            return [self.base_note + key_offset + root, self.base_note + key_offset + root + 7]
        if 'i' in chord_symbol:
            return [self.base_note + key_offset + root, self.base_note + key_offset + root + 3, self.base_note + key_offset + root + 7]
        return [self.base_note + key_offset + root, self.base_note + key_offset + root + 4, self.base_note + key_offset + root + 7]

    def snap_to_scale(self, note, scale):
        octave = note // 12
        degree = note % 12
        closest = min(scale, key=lambda x: abs(x - degree))
        return (octave * 12) + closest

    def note_to_midi_offset(self, key):
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        key_clean = key.replace(" Minor", "").replace(" Major", "").strip().upper()
        return note_map.get(key_clean, 0)

    def beats_to_ticks(self, beats, tempo, ticks_per_beat):
        return int(mido.second2tick(beats * (60 / tempo), ticks_per_beat, mido.bpm2tempo(tempo)))