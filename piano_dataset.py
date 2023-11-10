from pathlib import Path
import argparse
import soundfile as sf

def fix(folder: Path):
    counter = 0
    for file in folder.iterdir():
        print("counter:", counter)
        if file.suffix == ".ogg":
            data, samplerate = sf.read(file)
            new_filename = file.with_suffix(".wav")
            sf.write(new_filename, data, samplerate)
        counter += 1

def rename_files_in_folder(folder: Path, new_folder: Path):
    if not folder.is_dir():
        return
    for file in folder.iterdir():
        if not file.is_file():
            continue
        new_name = "piano-" + file.name
        file.rename(new_folder / new_name)

    folder.rmdir()

def configure_piano(overall_folder: Path):
    new_folder = overall_folder / "piano"
    if not new_folder.exists():
        new_folder.mkdir()

    for subfolder in sorted(overall_folder.iterdir()):
        if subfolder.name == "piano":
            fix(subfolder)
            continue
        rename_files_in_folder(subfolder, new_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('piano_folder', type=str)
    args = parser.parse_args()
    configure_piano(Path(args.piano_folder))
