import audio_preprocessing
import sklearn
import numpy as np
import pandas as pd
import os
from pathlib import Path
import re


### Get all training files into one large file after preprocessing
frame = pd.DataFrame()
path = Path('datasets/good-sounds/good-sounds/good-sounds/sound_files')
for folder in path.iterdir():
    for sound_file in folder.iterdir():
        # Process each sound file
        try:
            processed_file = audio_preprocessing.preprocessing(folder, sound_file)
        except Exception as e:
            print(e)
            continue
        # Add the expected output
        p = re.compile('\\\\[^sound_files][a-zA-z_a-zA-Z]*$')
        folder2 = p.search(str(folder)).group()
        folder2 = folder2[1:]
        processed_file.append(str(folder2))
        # Add the complete data instance to one large DataFrame
        frame = pd.concat([frame, pd.DataFrame([processed_file])], ignore_index=True)

print(frame)
frame.to_csv('training_data.csv', index=False)
### Train-test-split


### Train the model and validate
