import streamlit as st
from universal_melody_generator import UniversalMelodyGenerator, SCALE_MAP, SONG_STRUCTURES
import mido
import fluidsynth
import tempfile
import os

SOUNDFONT_PATH = 'soundfonts/FluidR3_GM.sf2'

fs = fluidsynth.Synth()
fs.start(driver="coreaudio")  # Use 'alsa' for Linux, 'dsound' for Windows
sfid = fs.sfload(SOUNDFONT_PATH)
fs.program_select(0, sfid, 0, 0)

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
        midi_file, structure = generator.generate_full_song(goal=mood, genre=genre)
        structure_display = " → ".join(structure)
    else:
        midi_file, tempo, key, mode, progression = generator.generate_melody_with_chords(goal=mood, genre=genre)
        structure_display = "Looped Phrase"

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mid')
    midi_file.save(tmp_file.name)
    st.session_state['midi_file_path'] = tmp_file.name

    st.success(f"Generated {generation_type} in {mood} mood!")
    st.write(f"**Genre:** {genre}")
    if generation_type == "Full Song":
        st.write(f"**Song Structure:** {structure_display}")
    else:
        st.write(f"**Tempo:** {tempo} BPM")
        st.write(f"**Key:** {key}")
        st.write(f"**Mode/Scale:** {mode.capitalize()}")
        st.write(f"**Chord Progression:** {progression}")
        st.write(f"**Structure:** {structure_display}")

if st.session_state['midi_file_path']:
    st.audio(st.session_state['midi_file_path'], format='audio/midi')

    if st.button("▶️ Play"):
        mid = mido.MidiFile(st.session_state['midi_file_path'])
        for msg in mid.play():
            if msg.type == 'program_change':
                fs.program_change(msg.channel, msg.program)
            elif msg.type == 'note_on':
                fs.noteon(msg.channel, msg.note, msg.velocity)
            elif msg.type == 'note_off':
                fs.noteoff(msg.channel, msg.note)
        fs.all_notes_off(0)

    if st.button("🗑️ Clear"):
        os.unlink(st.session_state['midi_file_path'])
        st.session_state['midi_file_path'] = None
        st.experimental_rerun()