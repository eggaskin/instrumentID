from pathlib import Path
import argparse

old_to_new_instrument_names = {
    'bass_alejandro_recordings': 'bass',
    'cello_margarita_attack': 'cello',
    'cello_margarita_dynamics_stability': 'cello',
    'cello_margarita_open_strings': 'cello',
    'cello_margarita_pitch_stability': 'cello',
    'cello_margarita_reference': 'cello',
    'cello_margarita_timbre_richness': 'cello',
    'cello_margarita_timbre_stability': 'cello',
    'cello_nico_improvement_recordings': 'cello',
    'clarinet_gener_evaluation_recordings': 'clarinet',
    'clarinet_gener_improvement_recordings': 'clarinet',
    'clarinet_marti_evaluation_recordings': 'clarinet',
    'clarinet_pablo_air': 'clarinet',
    'clarinet_pablo_attack': 'clarinet',
    'clarinet_pablo_dynamics_stability': 'clarinet',
    'clarinet_pablo_pitch_stability': 'clarinet',
    'clarinet_pablo_reference': 'clarinet',
    'clarinet_pablo_richness': 'clarinet',
    'clarinet_pablo_timbre_stability': 'clarinet',
    'clarinet_scale_gener_recordings': 'clarinet',
    'flute_almudena_air': 'flute',
    'flute_almudena_attack': 'flute',
    'flute_almudena_dynamics_change': 'flute',
    'flute_almudena_evaluation_recordings': 'flute',
    'flute_almudena_reference': 'flute',
    'flute_almudena_reference_piano': 'flute',
    'flute_almudena_stability': 'flute',
    'flute_almudena_timbre': 'flute',
    'flute_josep_evaluation_recordings': 'flute',
    'flute_josep_improvement_recordings': 'flute',
    'flute_scale_irene_recordings': 'flute',
    'oboe_marta_recordings': 'oboe',
    'piccolo_irene_recordings': 'piccolo',
    'sax_alto_scale_2_raul_recordings': 'sax_alto',
    'sax_alto_scale_raul_recordings': 'sax_alto',
    'sax_tenor_tenor_scales_2_raul_recordings': 'sax_tenor',
    'sax_tenor_tenor_scales_raul_recordings': 'sax_tenor',
    'saxo_bariton_raul_recordings': 'sax_baritone',
    'saxo_raul_recordings': 'sax_soprano',
    'saxo_soprane_raul_recordings': 'sax_soprano',
    'saxo_tenor_iphone_raul_recordings': 'sax_tenor',
    'saxo_tenor_raul_recordings': 'sax_tenor',
    'trumpet_jesus_evaluation_recordings': 'trumpet',
    'trumpet_jesus_improvement_recordings': 'trumpet',
    'trumpet_ramon_air': 'trumpet',
    'trumpet_ramon_attack_stability': 'trumpet',
    'trumpet_ramon_dynamics_stability': 'trumpet',
    'trumpet_ramon_evaluation_recordings': 'trumpet',
    'trumpet_ramon_pitch_stability': 'trumpet',
    'trumpet_ramon_reference': 'trumpet',
    'trumpet_ramon_timbre_stability': 'trumpet',
    'trumpet_scale_jesus_recordings': 'trumpet',
    'violin_laia_improvement_recordings': 'violin',
    'violin_laia_improvement_recordings_2': 'violin',
    'violin_raquel_attack': 'violin',
    'violin_raquel_dynamics_stability': 'violin',
    'violin_raquel_pitch_stability': 'violin',
    'violin_raquel_reference': 'violin',
    'violin_raquel_richness': 'violin',
    'violin_raquel_timbre_stability': 'violin',
    'violin_violin_scales_laia_recordings': 'violin'
}

def print_folders(folder: Path):
    for subfolder in sorted(folder.iterdir()):
        print(str(subfolder).split("/")[-1])


def merge_player_folders(instrument_folder):
    # Extract the recordings in each player folder into the outer instrument folder
    # Give the recordings a new name, a 6-digit number
    # Delete the player folders
    index = 0
    for player_folder in instrument_folder.iterdir():
        if not player_folder.is_dir():
            continue
        for recording_file in player_folder.iterdir():
            if not recording_file.is_file():
                continue
            index += 1
            new_name = instrument_folder / (str(index).zfill(6))
            recording_file.rename(new_name)
        player_folder.rmdir()


def configure_good_sounds(good_sounds_folder: Path):
    sound_files_folder = good_sounds_folder / 'sound_files'
    instruments = [
        'bass',
        'cello',
        'clarinet',
        'flute',
        'oboe',
        'piccolo',
        'sax_alto',
        'sax_tenor',
        'sax_baritone',
        'sax_soprano',
        'trumpet',
        'violin'
    ]
    combine_instrument_folders(instruments, sound_files_folder)
    for instrument in instruments:
        instrument_folder = sound_files_folder / instrument
        merge_player_folders(instrument_folder)


def combine_instrument_folders(instruments, sound_files_folder):
    # Combine all the files into the new instrument folders
    for instrument in instruments:
        instrument_folder = sound_files_folder / instrument
        print(instrument_folder)
        if not instrument_folder.exists():
            instrument_folder.mkdir()
        for old_instrument_name, new_instrument_name in old_to_new_instrument_names.items():
            if instrument == new_instrument_name:
                old_instrument_folder = sound_files_folder / old_instrument_name
                if not old_instrument_folder.exists():
                    continue
                # Only move folders, delete files
                for file in old_instrument_folder.iterdir():
                    new_name = instrument_folder / (old_instrument_name + "_" + file.name)
                    if file.is_dir():
                        file.rename(new_name)
                    else:
                        file.unlink()
                # Delete the old instrument folder
                old_instrument_folder.rmdir()


if __name__ == '__main__':
    # Use arg parser to get the good sounds folder
    parser = argparse.ArgumentParser(description='Configure good sounds')
    parser.add_argument('good_sounds_folder', type=str, help='The folder containing the good sounds')
    args = parser.parse_args()
    configure_good_sounds(Path(args.good_sounds_folder))