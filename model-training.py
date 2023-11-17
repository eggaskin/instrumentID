import audio_preprocessing
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import numpy as np
import pandas as pd
from pathlib import Path


### Get all training files into one large file after preprocessing
def createCSV():
    frame = pd.DataFrame()
    folder = Path('datasets/training')
    for sound_file in folder.iterdir():
        # Process each sound file
        try:
            processed_file = audio_preprocessing.preprocessing(folder, sound_file)
        except Exception as e:
            print(e)
            continue

        # Add the expected output
        instrument = str(sound_file).split('-')[0]
        processed_file.append(instrument)

        # Add the complete data instance to one large DataFrame
        frame = pd.concat([frame, pd.DataFrame([processed_file])], ignore_index=True)

    
    frame.to_csv('training_data.csv', index=False)


### Train-test-split
frame = pd.read_csv('training_data.csv')
enc = preprocessing.LabelEncoder()
y = frame.iloc[:,20]
y = y.values.reshape(-1, 1)
y = y.ravel()
y = enc.fit_transform(y)
X = frame.drop(frame.columns[-1], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


### Train the model and validate
try:
    clf = load('model.joblib')
except Exception as e:
    clf = MLPClassifier(hidden_layer_sizes=[25,25],activation='relu',max_iter=2000,warm_start=True)

# Train
clf.fit(X_train.values, y_train)

# Get accuracies, predictions, number of iterations, and best loss
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test.values, y_test)
predictions = pd.DataFrame(enc.inverse_transform(clf.predict(X_test.values)))
predictions['Actual'] = enc.inverse_transform(y_test)
best_loos = clf.best_loss_
num_iter = clf.n_iter_

# Put all the values into a table and write the tables to .csv files
table = pd.DataFrame({'Train Accuracy' : [train_acc], 'Test Accuracy' : [test_acc], 'Best Loss' : [best_loos], 'Number Interations' : [num_iter]})
predictions.to_csv('predictions.csv',index=False)
table.to_csv('model_results.txt', mode='a', index=False)

# Save the current state of the model
dump(clf, 'model.joblib')

# A single example prediction
test_X = [-749.64379883,   80.23860931,  -43.81538773,  -43.7645607,   -27.48980141,  -16.35895538,  -23.65286446, -19.15673065,  -13.72361755,   12.83826256,   36.4092598,   51.41631699,   39.2676506,    -0.98627788,  -27.72893715,  -40.71960068,  -22.84813309,   13.56567955,   16.81631088,    0.98919487]
test_y = y[-1]
print('Predicted: {0}'.format(clf.predict([test_X])) + "----Actual: {0}".format(test_y))
