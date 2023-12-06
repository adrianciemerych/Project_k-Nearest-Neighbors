# Project_k-Nearest-Neighbors

This is a k-Nearest Neighbors model function.
The function takes following parameters:
  - X_train, y_train - training data and target, which will be used to train the model
  - X_test - set using to check how model working on new data
  - k (default 3) - number of nearest neighbors taken into account
  - metrics (default 'Euclidean') - type of distance metric using by model ('Euclidean', 'Manhattan')
This function returns list with predictions for X_test data
