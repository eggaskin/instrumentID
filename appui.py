import streamlit as st
import librosa
from sklearn import preprocessing
import pandas as pd
import joblib
import numpy as np
from st_audiorec import st_audiorec
import os
import io
from audio_preprocessing import removeSilence, mel_spectogram_generator

st.title('Instrument ID')

st.write('\n\n')
st.write('Record around 5 seconds of audio by clicking Start Recording, then Stop.')
st.write('The audio will be displayed below, and a prediction below that!')

wav_audio_data = st_audiorec()

# add some spacing and informative messages
col_info, col_space = st.columns([0.57, 0.43])

if wav_audio_data is not None:
    # display audio data as received on the Python side
    #col_playback, col_space = st.columns([0.58,0.42])
    #with col_playback:
        #st.audio(wav_audio_data, format='audio/wav')

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
    dat = np.array(dat).reshape(1, -1)
    st.write(dat)

    # loading a model from pickle
    model = joblib.load('model.joblib')
    # using the loaded model to make predictions

    # only predict if dat does not contain nan
    if len(dat) >0: #dat != None: #not np.isnan(dat).any():

        pred = model.predict(dat)
        st.write("Your prediction:")
        st.write(pred)

        # read in classes from encoder_classes.csv and get index
        enc_classes = pd.read_csv("encoder_classes.csv")
        enc_classes = enc_classes.values.tolist()
        #st.write(enc_classes)
        enc_classes = [i[0] for i in enc_classes]
        st.write(enc_classes)
        st.write(pred[0], type(pred[0]))
        predclass = enc_classes[pred[0]]


        # get actual class
        #enc = preprocessing.LabelEncoder()
        #enc_classes = pd.DataFrame(enc.classes_)
        #enc_classes.to_csv("encoder_classes.csv", index=False)
        #predclass = enc.inverse_transform(pred)
        st.write("Your prediction class:")
        st.write(predclass)
    else: st.write("No data to predict...")

st.write('\n\n')
st.markdown('Using the audio recorder library, '
            
    '[st_audiorec](https://github.com/stefanrmmr/streamlit-audio-recorder)')