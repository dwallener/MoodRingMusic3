import mido
from mido import Message, MidiFile, MidiTrack
import random

# 🎵 Instrument Assignments by Genre (General MIDI Program Numbers)
GENRE_INSTRUMENTS = {
    "Classical": {
        "melody": 40,            # Violin
        "chords_primary": 41,    # Viola
        "chords_secondary": 42,  # Cello
        "bass": 43                # Contrabass
    },
    "Jazz": {
        "melody": 56,             # Trumpet
        "chords_primary": 0,      # Acoustic Grand Piano
        "chords_secondary": 24,   # Acoustic Guitar (Nylon)
        "bass": 33                # Acoustic Bass
    },
    "Pop": {
        "melody": 4,               # Electric Piano 1
        "chords_primary": 27,      # Electric Guitar (Jazz)
        "chords_secondary": 81,    # Lead 2 (Sawtooth Synth)
        "bass": 34                 # Electric Bass (Finger)
    },
    "Dance": {
        "melody": 80,              # Lead 1 (Square)
        "chords_primary": 81,      # Lead 2 (Sawtooth)
        "chords_secondary": 89,    # Pad 2 (Warm)
        "bass": 38                 # Synth Bass 1
    }
}

# 🎶 Available Scales
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10],
    'harmonic minor': [0, 2, 3, 5, 7, 8, 11],
    'pentatonic': [0, 2, 4, 7, 9]
}

# 🎯 Mood to Scale Mapping
SCALE_MAP = {
    "Focus": ["dorian", "mixolydian", "major"],
    "Relax": ["lydian", "major", "pentatonic"],
    "Energy": ["major", "mixolydian", "phrygian"],
    "Sleep": ["aeolian", "dorian", "pentatonic"],
    "Creative Flow": ["lydian", "mixolydian", "dorian"],
    "Calm Confidence": ["major", "dorian"],
    "Romantic": ["harmonic minor", "aeolian", "major"],
    "Reflective": ["aeolian", "dorian", "pentatonic"]
}

# 🎹 Mood to Key Mapping
KEY_MAP = {
    "Focus": ["C", "A", "F"],
    "Relax": ["F", "Eb", "Bb"],
    "Energy": ["G", "D", "E"],
    "Sleep": ["A", "C", "G"],
    "Creative Flow": ["E", "G", "D"],
    "Calm Confidence": ["C", "F", "Bb"],
    "Romantic": ["Bb", "D", "A"],
    "Reflective": ["C", "D", "Eb", "G"]
}

# 🎼 Chord Progressions per Mood
CHORD_PROGRESSIONS = {
    "Focus": ["I-vi-IV-V", "ii-V-I", "Modal Dorian"],
    "Relax": ["I-IV-I", "I-vi-IV-I", "Lydian Static"],
    "Energy": ["I-V-vi-IV", "IV-V-I", "Dominant Cycle"],
    "Sleep": ["I-ii-I", "Pedal Tones", "Open 5ths"],
    "Creative Flow": ["I-V-vi-IV", "ii-V-I", "Modal Mixolydian"],
    "Calm Confidence": ["I-IV-V-I", "ii-V-I", "I-vi-ii-V"],
    "Romantic": ["I-vi-IV-V", "ii-V-I", "I-IV-I"],
    "Reflective": ["I-vi-IV-I", "Modal Aeolian", "I-ii-IV-V"]
}

# 📚 Song Structures by Genre
SONG_STRUCTURES = {
    "Pop": ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus", "Outro"],
    "Dance": ["Intro", "Build", "Drop", "Break", "Drop", "Outro"],
    "Classical": ["Intro", "Theme", "Development", "Recapitulation", "Coda"],
    "Jazz": ["Head", "Solo", "Solo", "Head", "Outro"]
}

class UniversalMelodyGenerator:
    def __init__(self, base_note=60):
        self.base_note = base_note

    def generate_full_song(self, goal='Focus', genre='Classical'):
        structure = SONG_STRUCTURES.get(genre, SONG_STRUCTURES["Pop"])
        full_midi = MidiFile()
        tracks = [MidiTrack() for _ in range(4)]
        for t in tracks:
            full_midi.tracks.append(t)

        instrument_set = GENRE_INSTRUMENTS.get(genre, GENRE_INSTRUMENTS["Classical"])
        for idx, (track_name, channel) in enumerate(zip(["melody", "chords_primary", "chords_secondary", "bass"], range(4))):
            tracks[idx].append(Message('program_change', program=instrument_set[track_name], time=0, channel=channel))

        for section in structure:
            mid, tempo, key, mode, progression = self.generate_section(goal, genre, section)
            for i, track in enumerate(mid.tracks):
                for msg in track:
                    tracks[i].append(msg)

        return full_midi, structure

    def generate_section(self, goal, genre, section):
        base_tempo = random.randint(60, 140)
        tempo_mod = {"Intro": -10, "Verse": 0, "Chorus": +10, "Bridge": +5, "Outro": -5,
                     "Build": +15, "Drop": +20, "Break": -10, "Theme": 0, "Development": +5,
                     "Recapitulation": 0, "Coda": -10, "Head": 0, "Solo": +10}.get(section, 0)
        tempo = max(60, base_tempo + tempo_mod)

        scale_choice = random.choice(SCALE_MAP[goal])
        key_choice = random.choice(KEY_MAP[goal])
        progression_choice = random.choice(CHORD_PROGRESSIONS[goal])

        mode = scale_choice.lower()
        key_offset = self.note_to_midi_offset(key_choice)
        scale = SCALES[mode]

        contour = self.random_contour()
        rhythm = self.random_rhythm(len(contour))

        mid = MidiFile()
        tracks = [MidiTrack() for _ in range(4)]
        for t in tracks:
            mid.tracks.append(t)

        tempo_microsec = mido.bpm2tempo(tempo)
        for t in tracks:
            t.append(mido.MetaMessage('set_tempo', tempo=tempo_microsec))

        current_note = self.base_note + key_offset
        for interval, duration in zip(contour, rhythm):
            current_note += interval
            scale_note = self.snap_to_scale(current_note, scale)
            ticks = self.beats_to_ticks(duration, tempo, mid.ticks_per_beat)
            velocity = random.randint(60, 90)
            tracks[0].append(Message('note_on', note=scale_note, velocity=velocity, time=0, channel=0))
            tracks[0].append(Message('note_off', note=scale_note, velocity=velocity, time=ticks, channel=0))

        self.add_chords(tracks[1:], progression_choice, key_offset, tempo, mid.ticks_per_beat, sum(rhythm))
        return mid, tempo, key_choice, mode, progression_choice

    def generate_melody_with_chords(self, goal='Focus', genre='Classical'):
        return self.generate_section(goal, genre, "Loop")

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
            "Modal Dorian": ["i", "IV", "v", "i"],
            "Modal Mixolydian": ["I", "bVII", "IV", "I"],
            "Lydian Static": ["I", "II", "I", "II"],
            "Dominant Cycle": ["V", "I", "V", "I"],
            "Pedal Tones": ["I", "I", "I", "I"],
            "Open 5ths": ["I5", "V5", "IV5", "I5"],
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