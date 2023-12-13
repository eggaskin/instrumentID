import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# read in data
pred_df = pd.read_csv('data5_predictions.csv')

# Get a counter for each instrument. Ex: How many "violin" values are there?
pred_df_actual_count = Counter(pred_df['Actual'])

# list of lists with wrong prediction and truth
incorrect = []

# list of lists with wrong prediction and name of file, which has the name of the played instrument
actual_filename = []
for pred, actual, file_name in zip(pred_df['0'], pred_df['Actual'], pred_df['Filename']):
    if pred != actual:
        incorrect.append([pred, actual])
        actual_filename.append([pred, file_name])

print('Number of incorrect predictions:', len(incorrect))
print('Accuracy:', 1 - len(incorrect) / len(pred_df))

# get the number of each instrument that was classified incorrectly
incorrect_actual = [actual[1] for actual in incorrect]
incorrect_actual_count = Counter(incorrect_actual)

## create dictionaries that have the same key order
# order the incorrect_actual_count Counter object
rearranged_incorrect_actual_count = {key: incorrect_actual_count[key] for key in list(pred_df_actual_count.keys())}

# order the pred_df_actual_count Counter object
rearranged_pred_df_actual_count = {key: pred_df_actual_count[key] for key in list(pred_df_actual_count.keys())}

# dictionary of error rate for each instrument
errors_dict = {}
for instrument, count in rearranged_pred_df_actual_count.items():
    errors_dict[instrument] = rearranged_incorrect_actual_count[instrument] / count

# data frame in descending order by error rate
errors_df = pd.DataFrame(errors_dict.items(), columns=['Instrument', 'Error Rate']).sort_values(by='Error Rate', ascending=False)
errors_df

# Show Confusion Matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# get the actual and predicted values
y_true = pred_df['Actual']
y_pred = pred_df['0']

# create confusion matrix
columns = np.unique(y_true)
data = confusion_matrix(y_true, y_pred, labels=columns)

df_cm = pd.DataFrame(data, columns=columns, index=columns)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, cmap="Blues", annot=df_cm, annot_kws={"size": 11})  # font size
plt.xticks(rotation=80)
plt.show()
