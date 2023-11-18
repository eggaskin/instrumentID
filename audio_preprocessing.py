from IPython.display import Audio
from pathlib import Path
import librosa
import pydub
import simpleaudio
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import re


# A function to play back the audio
def playback(file_to_open):
    sound = pydub.AudioSegment.from_wav(file_to_open)
    playback = simpleaudio.play_buffer(
        sound.raw_data,
        num_channels=sound.channels,
        bytes_per_sample=sound.sample_width,
        sample_rate=sound.frame_rate
    )
    time.sleep(5)
    playback.stop()


# A function to remove silence
def removeSilence(signal):
    return signal[librosa.effects.split(signal)[0][0]: librosa.effects.split(signal)[0][-1]]


# A function to create same-sized chunks of audio
def create_chunks(audio, filename):
    if len(audio) < 10000:
        return "Audio too small"
    chunk_no = 0
    t1 = 0
    for step in range(0, len(audio), 3000):
        t2 = step
        if step == 0:
            t1 = step
            continue
        curr_chunk = audio[t1:t2]
        chunk_no += 1
        curr_chunk.export(out_f=filename + '-' + str(chunk_no) + '.wav', format="wav")
        t1, t2 = t2, t1


# A function to display a waveplot of the original audio
def displayWavePlot(signal, sample_rate):
    librosa.display.waveshow(signal, sr=sample_rate)
    plt.show()


# A function to generate and potentially display the mel-spectrogram
def mel_spectogram_generator(audio_name, signal, sample_rate, augmentation, target_path, create_figure):
    N_FFT = 1028
    HOP_SIZE = 256
    N_MELS = 128
    WIN_SIZE = 1024
    WINDOW_TYPE = 'hann'
    FEATURE = 'mel'
    FMIN = 1400

    if len(signal) < N_FFT:
        # Audio too small
        return

    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sample_rate, hop_length=HOP_SIZE, n_fft=N_FFT)
    spectrogram = np.abs(mel_signal)
    output_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    if create_figure:
        plt.figure(figsize=(8, 7))
        librosa.display.specshow(output_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel', cmap='magma',
                                 hop_length=HOP_SIZE)
        plt.colorbar(label='Amplitude')
        plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
        plt.xlabel('Time', fontdict=dict(size=15))
        plt.ylabel('Frequency', fontdict=dict(size=15))
        # plt.show()

        plt.savefig(os.path.join(target_path + augmentation + audio_name[:-4] + '.png'))
        plt.clf()
        plt.close("all")
    return np.mean(librosa.feature.mfcc(S=output_spectrogram, sr=sample_rate, n_mfcc=20).T, axis=0).tolist()


# The preprocessing function that should be called by other methods
def preprocessing(data_folder, file_to_open: Path):
    # Use regex to get the correct class name for the file
    instrument = file_to_open.name.split('-')[0]

    # Some of the files don't work so it is wrapped in a try-catch block
    try:
        signal, sample_rate = librosa.load(file_to_open, sr=None)
    except Exception as e:
        print("Preprocessing Exception: " + str(e))
        bad_sounds_folder = Path('datasets/bad-sounds')
        if not bad_sounds_folder.exists():
            os.mkdir(bad_sounds_folder)
        os.rename(file_to_open, bad_sounds_folder / file_to_open.name)
        raise Exception("Bad File: " + str(file_to_open))

    # If we ever need to create chunks in the future, this code will do it
    # create_chunks(signal, augmentation)

    # Remove silence from the file
    signal = removeSilence(signal)

    # Create the mel-spectrogram vector
    return (
        mel_spectogram_generator('out', signal, sample_rate, instrument, os.path.join(data_folder, 'output'), False))
