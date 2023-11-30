import os
from pathlib import Path
import argparse
import soundfile as sf


def configure_piano(apl_dataset_folder: Path, training_folder: Path):
    if not training_folder.exists():
        training_folder.mkdir(parents=True)

    for sound_file in training_folder.iterdir():
        if not sound_file.is_file() or sound_file.name.startswith('.'):
            continue
        if "piano_practice" in sound_file.name:
            os.remove(sound_file)

    index = 0
    for subfolder in sorted(apl_dataset_folder.iterdir()):
        if not subfolder.is_dir():
            continue
        for file in subfolder.iterdir():
            if file.suffix == ".ogg":
                data, samplerate = sf.read(file)
                new_filename = training_folder / ("piano" + "-piano_practice-" + str(index).zfill(6) + ".wav")
                print(new_filename)
                sf.write(new_filename, data, samplerate)
            index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('apl_dataset_folder', type=str)
    parser.add_argument('training_folder', type=str)
    args = parser.parse_args()
    configure_piano(Path(args.apl_dataset_folder), Path(args.training_folder))
