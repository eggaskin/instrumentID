import argparse
import os

from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier

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
    enc_classes = pd.DataFrame(enc.classes_)
    enc_classes.to_csv("encoder_classes.csv", index=False)
    X = frame.drop(frame.columns[-2:], axis=1)
    X_train, X_test, y_train, y_test, f_names_train, f_names_test = train_test_split(X, y, f_names, test_size=0.2, shuffle=True)

    # Ensure at least 50 samples per class in the test set.
    # If not, pull more from the training set
    for i in range(0, len(enc.classes_)):
        # reindex the sets
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        f_names_test = f_names_test.ravel()
        f_names_train = f_names_train.ravel()
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        if len(y_test[y_test == i]) < 50:
            # Get the number of samples needed
            num_samples_needed = 50 - len(y_test[y_test == i])
            # Get the indices of the samples needed
            indices = np.where(y_train == i)[0][:num_samples_needed]
            # Get the samples
            samples = X_train.iloc[indices]
            # Add the samples to the test set
            X_test = pd.concat([X_test, samples], ignore_index=True)
            y_test = np.append(y_test, y_train[indices])
            f_names_test = np.append(f_names_test, f_names_train[indices])
            # Remove the samples from the training set
            X_train = X_train.drop(index=indices)
            y_train = np.delete(y_train, indices)
            f_names_train = np.delete(f_names_train, indices)
            print(f"Class {i} has {len(y_test[y_test == i])} samples in the test set")
            print(f"Class {i} has {len(y_train[y_train == i])} samples in the training set")

    # Limit the number of samples per class in the test set to 50.
    # Move the rest to the training set
    for i in range(0, len(enc.classes_)):
        # reindex the sets
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        f_names_test = f_names_test.ravel()
        f_names_train = f_names_train.ravel()
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        if len(y_test[y_test == i]) > 50:
            # Get the number of samples needed
            num_samples_needed = len(y_test[y_test == i]) - 50
            # Get the indices of the samples needed
            indices = np.where(y_test == i)[0][:num_samples_needed]
            # Get the samples
            samples = X_test.iloc[indices]
            # Add the samples to the training set
            X_train = pd.concat([X_train, samples], ignore_index=True)
            y_train = np.append(y_train, y_test[indices])
            f_names_train = np.append(f_names_train, f_names_test[indices])
            # Remove the samples from the test set
            X_test = X_test.drop(index=indices)
            y_test = np.delete(y_test, indices)
            f_names_test = np.delete(f_names_test, indices)

        print(f"Class {i} has {len(y_test[y_test == i])} samples in the test set")
        print(f"Class {i} has {len(y_train[y_train == i])} samples in the training set")

    # Remove Classes with 0 examples in the training set
    for i in range(0, len(enc.classes_)):
        # reindex the sets
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        f_names_train = f_names_train.ravel()
        f_names_test = f_names_test.ravel()
        if len(y_train[y_train == i]) == 0:
            # Get the indices of the samples needed
            indices = np.where(y_test == i)[0]
            # Remove the samples from the test set
            X_test = X_test.drop(index=indices)
            y_test = np.delete(y_test, indices)
            f_names_test = np.delete(f_names_test, indices)
            print(f"Class {i} has {len(y_test[y_test == i])} samples in the test set")
            print(f"Class {i} has {len(y_train[y_train == i])} samples in the training set")

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    for i in range(0, len(enc.classes_)):
        if len(y_train[y_train == i]) == 0:
            continue
        while len(y_train[y_train == i]) < 8000:
            num_samples_needed = 8000 - len(y_train[y_train == i])
            X_train = pd.concat([X_train, X_train[y_train == i][:num_samples_needed]], ignore_index=True)
            f_names_train = np.append(f_names_train, f_names_train[y_train == i][:num_samples_needed])
            y_train = np.append(y_train, y_train[y_train == i][:num_samples_needed])
        print(f"Class {i} has {len(y_test[y_test == i])} samples in the test set")
        print(f"Class {i} has {len(y_train[y_train == i])} samples in the training set")



    # hid=(500, 500), alpha=0.001: test acc=0.89, train acc=0.989
    # hid=(500, 500), alpha=0.005: test acc=0.894, train acc=0.99
    # hid=(500, 500), alpha=0.010: test acc=0.896, train acc=0.986
    # hid=(500, 500), alpha=0.010: test acc=0.871, train acc=0.980 activation = 'tanh'
    # hid=(500, 500), alpha=0.010: test acc=0.881, train acc=0.952 early_stopping=True, validation_fraction=0.2
    # hid=(500, 500), alpha=0.005: test acc=0.886, train acc=0.961 early_stopping=True, validation_fraction=0.1
    # hid=(500, 500), alpha=0.020: test acc=0.890, train acc=0.983

    # hid=(300, 300), alpha=0.010: test acc=0.884, train acc=0.979
    # hid=(600, 600), alpha=0.020: test acc=0.888, train acc=0.979

    # hid=(400, 400), alpha=0.001: test acc=0.867, train acc=0.927
    # hid=(600, 600), alpha=0.005: test acc=0.879, train acc=0.964 early_stopping=True, validation_fraction=0.1

    # Best test accuracy model: hid=(500, 500), alpha=0.01: test acc=0.896, train acc=0.986
    best_params = {
        "hidden_layer_sizes": [1000, 1000],
        "activation": 'relu',
        "max_iter": 2000,
        "alpha": 0.01,
        "solver": 'adam',
        "warm_start": False
    }

    other_layers_sizes = [150, 200, 250, 300, 350, 400, 450, 550, 600, 650]
    ## Train the model and validate

    clf = MLPClassifier(hidden_layer_sizes=[500,500], max_iter=2000,alpha=0.01, solver='adam', warm_start=False, verbose=1)

    # Train
    clf.fit(X_train.values, y_train)

    # Get accuracies, predictions, number of iterations, and best loss
    clf_train_acc = clf.score(X_train, y_train)
    clf_test_acc = clf.score(X_test.values, y_test)
    clf_predictions = pd.DataFrame(enc.inverse_transform(clf.predict(X_test.values)))
    clf_predictions['Actual'] = enc.inverse_transform(y_test)
    clf_predictions['Filename'] = f_names_test.values
    clf_best_loos = clf.best_loss_
    clf_num_iter = clf.n_iter_

    # Put all the values into a table and write the tables to .csv files
    clf_table = pd.DataFrame({'Train Accuracy': [clf_train_acc], 'Test Accuracy': [clf_test_acc], 'Best Loss': [clf_best_loos], 'Number Iterations': [clf_num_iter]})
    clf_predictions.to_csv('clf_predictions.csv', index=False)
    clf_table.to_csv('clf_model_results.txt', mode='a', index=False)

    # Save the current state of the model
    dump(clf, 'clf_model.joblib')


    # TRAIN THE GMM
    if os.path.exists("G:\Documents\CS\CS201R\instrumentID\GMM_model.joblib"):
        GMM = load("G:\Documents\CS\CS201R\instrumentID\GMM_model.joblib")
    else:
        GMM = HistGradientBoostingClassifier(learning_rate=0.01, max_iter=2000, warm_start=True, verbose=1)

    # Train
    GMM.fit(X_train.values, y_train)

    # Get accuracies, predictions, number of iterations, and best loss
    GMM_train_acc = GMM.score(X_train, y_train)
    GMM_test_acc = GMM.score(X_test.values, y_test)
    GMM_predictions = pd.DataFrame(enc.inverse_transform(GMM.predict(X_test.values)))
    GMM_predictions['Actual'] = enc.inverse_transform(y_test)
    GMM_predictions['Filename'] = f_names_test.values
    GMM_num_iter = GMM.n_iter_

    # Put all the values into a table and write the tables to .csv files
    table = pd.DataFrame(
        {'Train Accuracy': [GMM_train_acc], 'Test Accuracy': [GMM_test_acc], 'Number Iterations': [GMM_num_iter]})
    GMM_predictions.to_csv('gmm_predictions.csv', index=False)
    table.to_csv('gmm_model_results.txt', mode='a', index=False)

    # Save the current state of the model
    dump(GMM, 'GMM_model.joblib')

    # TRAIN THE ADA
    if os.path.exists("G:\Documents\CS\CS201R\instrumentID\\ada_model.joblib"):
        ADA = load("G:\Documents\CS\CS201R\instrumentID\\ada_model.joblib")
    else:
        ADA = AdaBoostClassifier(learning_rate=0.01)

    # Train
    ADA.fit(X_train.values, y_train)

    # Get accuracies, predictions, number of iterations, and best loss
    ADA_train_acc = ADA.score(X_train, y_train)
    ADA_test_acc = ADA.score(X_test.values, y_test)
    ADA_predictions = pd.DataFrame(enc.inverse_transform(ADA.predict(X_test.values)))
    ADA_predictions['Actual'] = enc.inverse_transform(y_test)
    ADA_predictions['Filename'] = f_names_test.values

    # Put all the values into a table and write the tables to .csv files
    ada_table = pd.DataFrame(
        {'Train Accuracy': [ADA_train_acc], 'Test Accuracy': [ADA_test_acc]})
    ADA_predictions.to_csv('ada_predictions.csv', index=False)
    ada_table.to_csv('ada_model_results.txt', mode='a', index=False)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train Model')
    argparser.add_argument('-c', '--create_CSV', action="store_true", help='Is a new CSV File needed?')
    argparser.add_argument('training_folder', type=str, help='Path to training data')
    args = argparser.parse_args()
    main(bool(args.create_CSV), Path(args.training_folder))
