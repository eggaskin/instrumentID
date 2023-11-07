from pathlib import Path
import argparse
import pandas as pd

old_to_new_instrument_names = {
    'clarinet': 'clarinet',
    'distorted electric guitar': 'guitar_distorted',
    'female singer': 'singer',
    'flute': 'flute',
    'piano': 'piano',
    'tenor saxophone': 'sax_tenor',
    'trumpet': 'trumpet',
    'violin': 'violin'
}

def configure_medley_solos(medley_solos_folder, solos_csv):
    # Read the solos csv
    solos_df = pd.read_csv(solos_csv)
    # Create training, test, and validation folders
    for folder_name in ['training', 'test', 'validation']:
        folder = medley_solos_folder / folder_name
        if not folder.exists():
            folder.mkdir(parents=True)
        create_instrument_folders(folder, solos_df)

    move_files(medley_solos_folder, solos_df)


def move_files(medley_solos_folder, solos_df):
    # Move the files into the instrument folders
    for file in medley_solos_folder.iterdir():
        print(file)
        if not file.is_file():
            continue
        if file.name.startswith('.'):
            continue
        uuid = file.name.split('_')[-1].split('.')[0]
        split = file.name.split("_")[1].split("-")[0]
        instrument = solos_df[solos_df['uuid4'] == uuid]['instrument'].values[0]
        instrument = old_to_new_instrument_names[instrument]
        instrument_folder = medley_solos_folder / split / instrument
        new_name = instrument_folder / (instrument + "-" + uuid + ".wav")
        file.rename(new_name)


def create_instrument_folders(medley_solos_folder, solos_df):
    # Create the instrument folders
    instruments = solos_df['instrument'].unique()
    for instrument in instruments:
        instrument_folder = medley_solos_folder / old_to_new_instrument_names[instrument]
        if not instrument_folder.exists():
            instrument_folder.mkdir()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Configure medley solos')
    argparser.add_argument('medley_solos_folder', type=str, help='Path to the medley_solos folder')
    argparser.add_argument('solos_csv', type=str, help='Path to the solos csv file')
    args = argparser.parse_args()
    medley_solos_folder = Path(args.medley_solos_folder)
    solos_csv = Path(args.solos_csv)
    configure_medley_solos(medley_solos_folder, solos_csv)
