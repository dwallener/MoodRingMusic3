import pretty_midi
import random

def generate_chord_progression(progression_template, key_root=60):
    chord_intervals = {
        'I': [0, 4, 7],
        'ii': [2, 5, 9],
        'iii': [4, 7, 11],
        'IV': [5, 9, 12],
        'V': [7, 11, 14],
        'vi': [9, 12, 16],
    }
    return [
        [key_root + i for i in chord_intervals[chord]]
        for chord in progression_template
    ]

def generate_bass_line(chord_progression, duration_per_chord):
    return [chord[0] for chord in chord_progression], [duration_per_chord] * len(chord_progression)

def generate_rhythm_pattern(bars, bpm):
    rhythm = []
    time = 0.0
    beat_duration = 60 / bpm
    for _ in range(bars * 4):
        rhythm.append(('kick', time))
        time += beat_duration / 2
        rhythm.append(('snare', time))
        time += beat_duration / 2
    return rhythm

def create_multi_track_midi(melody_pitches, melody_durations, 
                            chord_progression_template, bpm=120, key_root=60):
    midi = pretty_midi.PrettyMIDI()

    # Melody Track
    melody_instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    start_time = 0.0
    for pitch, duration in zip(melody_pitches, melody_durations):
        melody_instrument.notes.append(
            pretty_midi.Note(velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration)
        )
        start_time += duration
    midi.instruments.append(melody_instrument)

    # Chord Progression
    chords = generate_chord_progression(chord_progression_template, key_root)

    # Harmony Track 1: Pad
    harmony_pad = pretty_midi.Instrument(program=48)  # Strings Ensemble
    start_time = 0.0
    for chord in chords:
        for note_pitch in chord:
            harmony_pad.notes.append(
                pretty_midi.Note(velocity=80, pitch=int(note_pitch), start=start_time, end=start_time + 2.0)
            )
        start_time += 2.0
    midi.instruments.append(harmony_pad)

    # Harmony Track 2: Arpeggiated
    harmony_arp = pretty_midi.Instrument(program=46)  # Harp or Arpeggio Sound
    start_time = 0.0
    arpeggio_spacing = 0.2  # Space between arpeggiated notes
    for chord in chords:
        t = start_time
        for note_pitch in chord:
            harmony_arp.notes.append(
                pretty_midi.Note(velocity=70, pitch=int(note_pitch), start=t, end=t + 0.5)
            )
            t += arpeggio_spacing
        start_time += 2.0
    midi.instruments.append(harmony_arp)

    # Bass Line
    bass_pitches, bass_durations = generate_bass_line(chords, 2.0)
    bass_instrument = pretty_midi.Instrument(program=32)  # Bass
    start_time = 0.0
    for pitch, duration in zip(bass_pitches, bass_durations):
        bass_instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch - 24), start=start_time, end=start_time + duration)
        )
        start_time += duration
    midi.instruments.append(bass_instrument)

    # Rhythm Track (Drums)
    rhythm_pattern = generate_rhythm_pattern(len(chords), bpm)
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
    for hit, time in rhythm_pattern:
        pitch = 36 if hit == 'kick' else 38  # MIDI Drum Map: 36 = Bass Drum, 38 = Snare
        drum_instrument.notes.append(
            pretty_midi.Note(velocity=110, pitch=pitch, start=time, end=time + 0.1)
        )
    midi.instruments.append(drum_instrument)

    return midi

# Example Usage
melody_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
melody_durations = [0.5] * len(melody_pitches)
chord_progression_template = ['I', 'V', 'vi', 'IV'] * 2  # 8 Bars

midi = create_multi_track_midi(melody_pitches, melody_durations, chord_progression_template)
midi.write("multi_track_output.mid")


