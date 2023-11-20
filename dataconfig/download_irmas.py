import argparse
import subprocess
import time
import zipfile
from pathlib import Path


def download_irmas(irmas_folder: Path):
    # check if zip file exists
    zip_location = irmas_folder / "IRMAS-TrainingData.zip"
    if zip_location.exists():
        print("IRMAS zip file already exists")
        return

    subprocess.run([
        "curl",
        "https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1",
        "-o",
        str(zip_location)])


def unzip_irmas(irmas_folder: Path):
    # unzip IRMAS dataset
    print("Unzipping IRMAS dataset...")
    t0 = time.time()
    with zipfile.ZipFile(irmas_folder / "IRMAS-TrainingData.zip", 'r') as zip_ref:
        zip_ref.extractall(irmas_folder)
    print("IRMAS dataset unzipped in {} seconds".format(time.time() - t0))


def main(irmas_folder: Path):
    download_irmas(irmas_folder)
    unzip_irmas(irmas_folder)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Configure IRMAS')
    argparser.add_argument('irmas_folder', type=str, help='Path to the IRMAS folder')
    args = argparser.parse_args()
    main(Path(args.irmas_folder))