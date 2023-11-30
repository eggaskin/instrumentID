import os
from pathlib import Path
import soundfile
import argparse


def copy_sound_files(sound_files_folder: Path, training_folder: Path):
    for sound_file in training_folder.iterdir():
        if not sound_file.is_file() or sound_file.name.startswith('.'):
            continue
        if "uiowa" in sound_file.name:
            os.remove(sound_file)
    index = 0
    for instrument_folder in sound_files_folder.iterdir():
        if not instrument_folder.is_dir():
            continue
        instrument = instrument_folder.stem
        for recording_file in sorted(instrument_folder.iterdir()):
            if not recording_file.is_file():
                continue
            if not recording_file.suffix == '.aiff':
                continue
            index += 1
            new_name = training_folder / (instrument + "-uiowa-" + str(index).zfill(6) + ".wav")

            print(new_name)
            # convert from aiff to wav
            sound_file = soundfile.SoundFile(recording_file)
            original_samplerate = sound_file.samplerate
            nd_array = sound_file.read(dtype='float32')
            if nd_array.ndim == 2:
                nd_array = nd_array[0] + nd_array[1]
            nd_array = nd_array / max(nd_array)
            soundfile.write(new_name, nd_array, samplerate=original_samplerate)


def configure_uiowa(datasets_folder: Path):
    sound_files_folder = datasets_folder / 'uiowa'
    training_folder = datasets_folder / 'training'
    if not sound_files_folder.exists():
        raise Exception("uiowa folder does not exist at" + str(sound_files_folder))
    if not training_folder.exists():
        training_folder.mkdir(parents=True)
    copy_sound_files(sound_files_folder, training_folder)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Configure UIOWA')
    argparser.add_argument('datasets_folder', type=str, help='Path to the datasets folder')
    args = argparser.parse_args()
    configure_uiowa(Path(args.datasets_folder))
