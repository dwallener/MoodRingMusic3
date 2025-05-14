import streamlit as st
from universal_melody_generator import UniversalMelodyGenerator, SCALE_MAP, SONG_STRUCTURES
import mido
import fluidsynth
import tempfile
import os
import threading

SOUNDFONT_PATH = 'soundfonts/FluidR3_GM.sf2'

fs = fluidsynth.Synth()
fs.start(driver="coreaudio")
sfid = fs.sfload(SOUNDFONT_PATH)
fs.program_select(0, sfid, 0, 0)

playback_event = threading.Event()

st.title("🎵 Universal Melody Generator")
st.subheader("Generate and Play Music by Mood")

mood = st.selectbox("Select Mood", list(SCALE_MAP.keys()))
st.subheader("Select Genre")
genre = st.radio("", ["Classical", "Jazz", "Pop", "Dance"], horizontal=True)

generation_type = st.radio("Select Generation Type", ["Loop", "Full Song"], horizontal=True)
generate_button = st.button("Generate MIDI")

if 'midi_file_path' not in st.session_state:
    st.session_state['midi_file_path'] = None

if generate_button:
    generator = UniversalMelodyGenerator()
    if generation_type == "Full Song":
        midi_file, structure, tempo, key, mode, progression = generator.generate_full_song(goal=mood, genre=genre)
        structure_display = " → ".join(structure)
    else:
        midi_file, tempo, key, mode, progression = generator.generate_melody_with_chords(goal=mood, genre=genre)
        structure_display = "Looped Phrase"

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mid')
    midi_file.save(tmp_file.name)
    st.session_state['midi_file_path'] = tmp_file.name

    # Persist generation parameters
    st.session_state['structure_display'] = structure_display
    st.session_state['tempo'] = tempo
    st.session_state['key'] = key
    st.session_state['mode'] = mode
    st.session_state['progression'] = progression
    st.session_state['genre'] = genre
    st.session_state['mood'] = mood

# 🎯 Always display last generated parameters if available
if st.session_state['midi_file_path']:
    st.audio(st.session_state['midi_file_path'], format='audio/midi')

    st.success(f"Generated {generation_type} in {st.session_state.get('mood', '')} mood!")
    st.write(f"**Genre:** {st.session_state.get('genre', '')}")
    st.write(f"**Structure:** {st.session_state.get('structure_display', '')}")
    st.write(f"**Tempo:** {st.session_state.get('tempo', '')} BPM")
    st.write(f"**Key:** {st.session_state.get('key', '')}")
    st.write(f"**Mode/Scale:** {st.session_state.get('mode', '').capitalize()}")
    st.write(f"**Chord Progression:** {st.session_state.get('progression', '')}")

    def play_midi(path):
        mid = mido.MidiFile(path)
        playback_event.clear()
        for msg in mid.play():
            if playback_event.is_set():
                fs.all_notes_off(0)
                break
            if msg.type == 'program_change':
                fs.program_change(msg.channel, msg.program)
            elif msg.type == 'note_on':
                fs.noteon(msg.channel, msg.note, msg.velocity)
            elif msg.type == 'note_off':
                fs.noteoff(msg.channel, msg.note)
        fs.all_notes_off(0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Play"):
            threading.Thread(target=play_midi, args=(st.session_state['midi_file_path'],), daemon=True).start()

    with col2:
        if st.button("⏹️ Stop Playing"):
            playback_event.set()

    if st.button("🗑️ Clear"):
        os.unlink(st.session_state['midi_file_path'])
        st.session_state['midi_file_path'] = None
        # Clear parameter persistence
        for key in ['structure_display', 'tempo', 'key', 'mode', 'progression', 'genre', 'mood']:
            st.session_state.pop(key, None)
        st.rerun()