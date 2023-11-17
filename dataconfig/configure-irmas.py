import shutil
import time
from pathlib import Path
import argparse

instruments = {"cel": 'cello', 'cla': 'clarinet', 'flu': 'flute', 'gac': 'guitar', 'gel': 0, 'org': 'organ',
               'pia': 'piano', 'sax': 'saxophone', 'tru': 'trumpet', 'vio': 'violin', 'voi': 'voice'}


def main(irmas_folder: Path, training_folder: Path):
    # rename folders as instrument names
    print("Renaming folders...")
    t0 = time.time()
    index = 0
    for folder in (irmas_folder / "IRMAS-TrainingData").iterdir():
        print(folder)
        if folder.name in instruments:
            for recording_file in folder.iterdir():
                if not recording_file.is_file():
                    continue
                if recording_file.name.startswith('.'):
                    continue
                index += 1
                new_name = training_folder / (instruments[folder.name] + "-irmas-" + str(index) + ".wav")
                print(new_name)
                shutil.copy(recording_file, new_name)
    print("Folders renamed in {} seconds".format(time.time() - t0))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Configure IRMAS')
    argparser.add_argument('irmas_folder', type=str, help='Path to the IRMAS folder')
    argparser.add_argument('training_folder', type=str, help='The folder to move the sound files to')
    args = argparser.parse_args()
    main(Path(args.irmas_folder), Path(args.training_folder))
