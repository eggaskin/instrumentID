import requests
import time
import zipfile
import os

path = "https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1"
HOME_DIRECTORY = "C:/Users/evely/Desktop/cs270/instrumentID/datasets/" # path to instrumentID folder
instruments = {"cel":'cello', 'cla':'clarinet', 'flu':'flute', 'gac':'guitar', 'gel':0, 'org':'organ', 'pia':'piano', 'sax':'saxophone', 'tru':'trumpet', 'vio':'violin', 'voi':'voice'}

# download IRMAS dataset
print("Downloading IRMAS dataset...")
t0 = time.time()
r = requests.get(path)
with open(HOME_DIRECTORY + "IRMAS-TrainingData.zip", 'wb') as f:
    f.write(r.content)
print("IRMAS dataset downloaded in {} seconds".format(time.time() - t0))

# unzip IRMAS dataset
print("Unzipping IRMAS dataset...")
t0 = time.time()
with zipfile.ZipFile(HOME_DIRECTORY + "IRMAS-TrainingData.zip", 'r') as zip_ref:
    zip_ref.extractall(HOME_DIRECTORY)
print("IRMAS dataset unzipped in {} seconds".format(time.time() - t0))

# delete zip file
print("Deleting IRMAS zip file...")
t0 = time.time()
os.remove(HOME_DIRECTORY + "IRMAS-TrainingData.zip")
print("IRMAS zip file deleted in {} seconds".format(time.time() - t0))

# rename folders as instrument names
print("Renaming folders...")
t0 = time.time()
for folder in os.listdir(HOME_DIRECTORY + "IRMAS-TrainingData/"):
    print(folder)
    if folder in instruments:
        os.rename(HOME_DIRECTORY + "IRMAS-TrainingData/" + folder, HOME_DIRECTORY + "IRMAS-TrainingData/" + instruments[folder])
    #else:
        # delete directory
        #os.rmdir(HOME_DIRECTORY + "IRMAS-TrainingData/" + folder)
print("Folders renamed in {} seconds".format(time.time() - t0))

