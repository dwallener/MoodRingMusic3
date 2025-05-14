import streamlit as st
from universal_melody_generator import UniversalMelodyGenerator, SCALE_MAP
import mido
import tempfile
import os
import subprocess
from pydub import AudioSegment

SOUNDFONT_PATH = 'soundfonts/FluidR3_GM.sf2'

st.title("🎵 Universal Melody Generator")
st.subheader("Generate and Play Music by Mood")

mood = st.selectbox("Select Mood", list(SCALE_MAP.keys()))
st.subheader("Select Genre")
genre = st.radio("", ["Classical", "Jazz", "Pop", "Dance"], horizontal=True)
generation_type = st.radio("Select Generation Type", ["Full Song"], horizontal=True)
continuous = st.toggle("Continuous Playback", value=False)
loop_duration_sec = st.slider("Loop Duration (seconds)", 15, 300, 60, 5)

if 'audio_file_path' not in st.session_state:
    st.session_state['audio_file_path'] = None

def generate_and_convert(fade_out_ms=2000):
    generator = UniversalMelodyGenerator()
    midi_file, structure, tempo, key, mode, progression = generator.generate_full_song(goal=mood, genre=genre)

    tmp_midi = tempfile.NamedTemporaryFile(delete=False, suffix='.mid')
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    midi_file.save(tmp_midi.name)

    subprocess.run([
        "fluidsynth", "-ni", SOUNDFONT_PATH, tmp_midi.name,
        "-F", tmp_wav.name, "-r", "44100"
    ], check=True)

    audio = AudioSegment.from_wav(tmp_wav.name)
    audio = audio.fade_out(fade_out_ms)
    audio.export(tmp_wav.name, format="wav")

    st.session_state['audio_file_path'] = tmp_wav.name
    st.session_state['tempo'] = tempo
    st.session_state['key'] = key
    st.session_state['mode'] = mode
    st.session_state['progression'] = progression
    st.session_state['structure'] = structure

if st.button("Generate & Play") or (continuous and not st.session_state['audio_file_path']):
    generate_and_convert()

if st.session_state['audio_file_path']:
    st.audio(st.session_state['audio_file_path'], format='audio/wav')

    # Parameter Display
    st.subheader("🎼 Current Song Parameters")
    if 'structure' in st.session_state:
        st.write(f"**Structure:** {' → '.join(st.session_state['structure'])}")
    st.write(f"**Tempo:** {st.session_state.get('tempo', '')} BPM")
    st.write(f"**Key:** {st.session_state.get('key', '')}")
    st.write(f"**Mode/Scale:** {st.session_state.get('mode', '')}")
    st.write(f"**Chord Progression:** {st.session_state.get('progression', '')}")

    # Auto-Playback & Auto-Reload via JS Timer
    loop_ms = loop_duration_sec * 1000
    st.markdown(f"""
        <script>
        window.onload = function() {{
            const audio = document.querySelector('audio');
            if (audio) {{
                audio.play();
            }}
            setTimeout(function() {{
                window.location.reload();
            }}, {loop_ms});
        }}
        </script>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear"):
        os.unlink(st.session_state['audio_file_path'])
        st.session_state['audio_file_path'] = None
        for param in ['tempo', 'key', 'mode', 'progression', 'structure']:
            st.session_state.pop(param, None)
        st.rerun()