import shutil
from pathlib import Path
import argparse
import os

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
    'sax_alto_scale_2_raul_recordings': 'saxophone',
    'sax_alto_scale_raul_recordings': 'saxophone',
    'sax_tenor_tenor_scales_2_raul_recordings': 'saxophone',
    'sax_tenor_tenor_scales_raul_recordings': 'saxophone',
    'saxo_bariton_raul_recordings': 'saxophone',
    'saxo_raul_recordings': 'saxophone',
    'saxo_soprane_raul_recordings': 'saxophone',
    'saxo_tenor_iphone_raul_recordings': 'saxophone',
    'saxo_tenor_raul_recordings': 'saxophone',
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


def copy_sound_files(sound_files_folder, training_folder):
    for sound_file in training_folder.iterdir():
        if not sound_file.is_file() or sound_file.name.startswith('.'):
            continue
        if "good-sounds" in sound_file.name:
            os.remove(sound_file)
    index = 0
    for old_instrument_name, new_instrument_name in old_to_new_instrument_names.items():
        old_instrument_folder = sound_files_folder / old_instrument_name
        if not old_instrument_folder.exists():
            continue
        for player_folder in old_instrument_folder.iterdir():
            if not player_folder.is_dir():
                continue
            for recording_file in player_folder.iterdir():
                if not recording_file.is_file():
                    continue
                index += 1
                new_name = training_folder / (new_instrument_name + "-good_sounds-" + str(index).zfill(6) + ".wav")
                print(new_name)
                shutil.copy(recording_file, new_name)


def configure_good_sounds(good_sounds_folder: Path, training_folder: Path):
    sound_files_folder = good_sounds_folder / 'sound_files'
    if not sound_files_folder.exists():
        raise Exception("sound_files folder does not exist at" + str(sound_files_folder))
    if not training_folder.exists():
        training_folder.mkdir(parents=True)
    copy_sound_files(sound_files_folder, training_folder)


if __name__ == '__main__':
    # Use arg parser to get the good sounds folder
    parser = argparse.ArgumentParser(description='Configure good sounds')

    parser.add_argument('good_sounds_folder', type=str, help='The folder containing the good sounds')
    parser.add_argument('training_folder', type=str, help='The folder to move the sound files to')

    args = parser.parse_args()
    configure_good_sounds(Path(args.good_sounds_folder), Path(args.training_folder))
