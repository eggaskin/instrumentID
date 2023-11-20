import argparse

import audio_preprocessing
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

### Get all training files into one large file after preprocessing
def createCSV(trainingPath: Path):
    print("Creating CSV")
    frame = pd.DataFrame()
    folder = trainingPath
    sound_files = list(folder.iterdir())
    for sound_file in tqdm(sound_files, dynamic_ncols=True, colour='blue'):
        # Process each sound file
        try:
            processed_file = audio_preprocessing.preprocessing(folder, sound_file)
        except Exception as e:
            tqdm.write(str(e))
            continue
        if not processed_file:
            continue

        # Add the expected output
        instrument = str(sound_file.name).split('-')[0]
        processed_file.append(instrument)
        processed_file.append(sound_file.name)

        # Add the complete data instance to one large DataFrame
        frame = pd.concat([frame, pd.DataFrame([processed_file])], ignore_index=True)
    
    frame.to_csv('training_data.csv', index=False)

def main(create_CSV: bool, training_data_path: Path):
    if create_CSV:
        createCSV(training_data_path)

    ### Train-test-split
    frame = pd.read_csv('training_data.csv')
    enc = preprocessing.LabelEncoder()
    y = frame.iloc[:, -2]
    f_names = frame.iloc[:, -1]
    y = y.values.reshape(-1, 1)
    y = y.ravel()
    y = enc.fit_transform(y)
    X = frame.drop(frame.columns[-2:], axis=1)
    X_train, X_test, y_train, y_test, f_names_train, f_names_test = train_test_split(X, y, f_names, test_size=0.2, shuffle=True)


    ### Train the model and validate
    try:
        clf = load('triple_model.joblib')
    except Exception as e:
        clf = MLPClassifier(hidden_layer_sizes=[25, 25], activation='relu',max_iter=2000, warm_start=False)

    # Train
    clf.fit(X_train.values, y_train)

    # Get accuracies, predictions, number of iterations, and best loss
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test.values, y_test)
    predictions = pd.DataFrame(enc.inverse_transform(clf.predict(X_test.values)))
    predictions['Actual'] = enc.inverse_transform(y_test)
    predictions['Filename'] = f_names_test.values
    best_loos = clf.best_loss_
    num_iter = clf.n_iter_

    # Put all the values into a table and write the tables to .csv files
    table = pd.DataFrame({'Train Accuracy': [train_acc], 'Test Accuracy': [test_acc], 'Best Loss': [best_loos], 'Number Iterations': [num_iter]})
    predictions.to_csv('triple_predictions.csv', index=False)
    table.to_csv('triple_model_results.txt', mode='a', index=False)

    # Save the current state of the model
    dump(clf, 'triple_model.joblib')

    # A single example prediction
    test_X = [-749.64379883,   80.23860931,  -43.81538773,  -43.7645607,   -27.48980141,  -16.35895538,  -23.65286446, -19.15673065,  -13.72361755,   12.83826256,   36.4092598,   51.41631699,   39.2676506,    -0.98627788,  -27.72893715,  -40.71960068,  -22.84813309,   13.56567955,   16.81631088,    0.98919487]
    test_y = y[-1]
    print('Predicted: {0}'.format(clf.predict([test_X])) + "----Actual: {0}".format(test_y))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train Model')
    argparser.add_argument('-c', '--create_CSV', action="store_true", help='Is a new CSV File needed?')
    argparser.add_argument('training_folder', type=str, help='Path to training data')
    args = argparser.parse_args()
    main(bool(args.create_CSV), Path(args.training_folder))
