import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# This is a k-Nearest Neighbors model function.
# The function takes following parameters:
# X_train, y_train - training data and target, which will be used to train the model
# X_test - set using to check how model working on new data
# k (default 3) - number of nearest neighbors taken into account
# metrics (default 'Euclidean') - type of distance metric using by model ('Euclidean', 'Manhattan')
# This function returns list with predictions for X_test data


def kNN (X_train, y_train, X_test, k=3, metrics ='Euclidean'):
    # Creating list for X_test predictions
    predictions = []
    # For each row from test variables, passing through all single row from train variables
    # and computing distance between 2 points:
    # - first: coordinates are created with train data (1 column = 1D, 2 columns = 2D, 5 columns = 5D...)
    # - second: coordinates are created with test data (number of columns should be equal to train data)
    # Function will adjust to n-dimensional data using loop that passing through all, i.e. n columns.
    # In the next step there is a conditions 'if' and 'elif', which determine particular metrics
    for test_row in range(len(X_test)):
        list_of_distances = []
        for train_row in range(len(X_train)):
            sum = 0
            if metrics == 'Euclidean':
                for column in range(len(X_train.columns)):
                    sum += (X_test.iloc[test_row][column] - X_train.iloc[train_row][column]) ** 2
                d = np.sqrt(sum)
            elif metrics == 'Manhattan':
                for column in range(len(X_train.columns)):
                    sum += abs(X_test.iloc[test_row][column] - X_train.iloc[train_row][column])
                d = sum
            else:
                print("Wrong parameter 'metrics'. Required 'Euclidean' or 'Manhattan'.")
                break
            list_of_distances.append(d)
        # Calculated distance between points is appending to list.
        # Next step is creating a DataFrame with distance for each train target
        distance_to_target = pd.DataFrame({'target' : list(y_train),
                                           'distance' : list_of_distances})
        # Sort values by distance in ascending order and using method 'head' for choose the
        # k smallest (k nearest neighbors)
        distance_to_target = distance_to_target.sort_values('distance').head(k)
        # Based on target count, using max() function, we are choosing the most often
        # occurs target. This target is assigning to predictions list.
        # After iterate all test_rows, function returns fully completed list with
        # predictions
        predictions.append(max(set(distance_to_target['target']),
                               key = list(distance_to_target['target']).count))
    return predictions


# Testing on iris data

# import dataset
raw_data = load_iris()

# Preparing data to split
data = raw_data['data']
target = raw_data['target']

all_data = np.c_[data, target]
df = pd.DataFrame(all_data, columns = raw_data['feature_names'] + ['target'])

data = df.copy()
target = data.pop('target')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, stratify = target)

# kNN function
y_pred = kNN(X_train, y_train, X_test, k = 3, metrics ='Manhattan')
print('Predictions: \n', y_pred)

print('True targets: \n', list(y_test))

# Checking accuracy
number = 0
for i in range(len(y_pred)):
    if y_pred[i] == list(y_test)[i]:
        number += 1
accuracy = round(number / len(y_test) * 100, 2)
print('Accuracy:', accuracy, '%')