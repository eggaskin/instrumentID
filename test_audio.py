from audio_preprocessing import *
import os
import random
import librosa
from IPython.display import Audio
import soundfile as sf

folder = Path("testing_data")
files = os.listdir(folder)
file = random.choice(files)



signal, sample_rate = librosa.load("testing_data/" + file, sr=None)

mel_spectogram_generator('out', signal, sample_rate, "test", os.path.join(folder, 'output1'), True)

signal = addWhiteNoise(signal)
Audio(data=signal, rate=int(sample_rate))
mel_spectogram_generator('out', signal, sample_rate, "test", os.path.join(folder, 'output2'), True)
sf.write('output.wav', signal, int(sample_rate), subtype='PCM_24')

