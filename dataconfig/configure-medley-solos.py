import os
import shutil
from pathlib import Path
import argparse
import pandas as pd

old_to_new_instrument_names = {
    'clarinet': 'clarinet',
    'distorted electric guitar': 'guitar_electric',
    'female singer': 'voice',
    'flute': 'flute',
    'piano': 'piano',
    'tenor saxophone': 'saxophone',
    'trumpet': 'trumpet',
    'violin': 'violin'
}


def copy_sound_files(medley_solos_folder, solos_df, training_folder):
    for sound_file in training_folder.iterdir():
        if not sound_file.is_file() or sound_file.name.startswith('.'):
            continue
        if "medley" in sound_file.name:
            os.remove(sound_file)
    # Move the files into the instrument folders
    index = 0
    for recording_file in sorted(medley_solos_folder.iterdir()):
        if not recording_file.is_file():
            continue
        if recording_file.name.startswith('.'):
            continue
        index += 1
        uuid = recording_file.name.split('_')[-1].split('.')[0]
        instrument = solos_df[solos_df['uuid4'] == uuid]['instrument'].values[0]
        instrument = old_to_new_instrument_names[instrument]
        new_name = training_folder / (instrument + "-medley_solos-" + str(index).zfill(6) + ".wav")
        print(new_name)
        shutil.copy(recording_file, new_name)


def configure_medley_solos(medley_solos_folder, solos_csv, training_folder):
    # Read the solos csv
    solos_df = pd.read_csv(solos_csv)
    if not training_folder.exists():
        training_folder.mkdir(parents=True)
    copy_sound_files(medley_solos_folder, solos_df, training_folder)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Configure medley solos')
    argparser.add_argument('medley_solos_folder', type=str, help='Path to the medley_solos folder')
    argparser.add_argument('solos_csv', type=str, help='Path to the solos csv file')
    argparser.add_argument('training_folder', type=str, help='The folder to move the sound files to')
    args = argparser.parse_args()
    configure_medley_solos(
        Path(args.medley_solos_folder),
        Path(args.solos_csv),
        Path(args.training_folder))
