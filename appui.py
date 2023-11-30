import streamlit as st
import librosa
import joblib
import numpy as np
from st_audiorec import st_audiorec
import os
import io
from audio_preprocessing import removeSilence, mel_spectogram_generator

st.title('Instrument ID')
st.markdown('Using the audio recorder library'
            
    '[GitHub](https://github.com/stefanrmmr/streamlit-audio-recorder)')
st.write('\n\n')

wav_audio_data = st_audiorec()

# add some spacing and informative messages
col_info, col_space = st.columns([0.57, 0.43])

if wav_audio_data is not None:
    # display audio data as received on the Python side
    col_playback, col_space = st.columns([0.58,0.42])
    with col_playback:
        st.audio(wav_audio_data, format='audio/wav')

    # Convert arrayBuffer to NumPy array
    wav_data = np.frombuffer(wav_audio_data, dtype=np.int16)
    wav_file_obj = io.BytesIO(wav_data.tobytes())

    # Perform audio processing with librosa
    try:
        signal, sr = librosa.load(wav_file_obj, sr=None)
    except Exception as e:
        st.write(str(e))
        st.write("Error in librosa.load()")
        st.stop()

    # Remove silence
    signal = removeSilence(signal)
    dat = mel_spectogram_generator('out', signal, sr, "", os.path.join("/", 'output'), False)

    # loading a model from pickle
    model = joblib.load('model.joblib')
    # using the loaded model to make predictions
    pred = model.predict(dat)
    st.write(pred)
