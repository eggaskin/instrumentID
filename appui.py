import streamlit as st
#import pickle
import pandas as pd
import sklearn as sk

from st_audiorec import st_audiorec

st.write("hi there")

st.title('streamlit audio recorder')
st.markdown('Implemented by '
    '[Stefan Rummer](https://www.linkedin.com/in/stefanrmmr/) - '
    'view project source code on '
            
    '[GitHub](https://github.com/stefanrmmr/streamlit-audio-recorder)')
st.write('\n\n')

# TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
# by calling this function an instance of the audio recorder is created
# once a recording is completed, audio data will be saved to wav_audio_data

wav_audio_data = st_audiorec() # tadaaaa! yes, that's it! :D

# add some spacing and informative messages
col_info, col_space = st.columns([0.57, 0.43])
with col_info:
    st.write('\n')  # add vertical spacer
    st.write('\n')  # add vertical spacer
    st.write('The .wav audio data, as received in the backend Python code,'
                ' will be displayed below this message as soon as it has'
                ' been processed. [This informative message is not part of'
                ' the audio recorder and can be removed easily] 🎈')

if wav_audio_data is not None:
    # display audio data as received on the Python side
    col_playback, col_space = st.columns([0.58,0.42])
    with col_playback:
        st.audio(wav_audio_data, format='audio/wav')

# saving a clf model to pickle
#pickle.dump(clf, open('model.pkl','wb'))
#TODO:
# loading a model from pickle
#joblib.load('model.joblib')
#model = pickle.load(open('model.pkl','rb'))
# using the loaded model to make predictions
#model.predict(data)

#stt_button = Button(label="Record an Instrument", width=100)

#stt_button.js_on_event("button_click", CustomJS(code="""CODE HERE"""))
