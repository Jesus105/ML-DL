# Data Preprocessing for Machine Learning
# This script demonstrates the steps of data preprocessing for a machine learning model. It includes loading data,
# shuffling the dataset, splitting features from labels, and creating training and test datasets.
# Author: Jesús Méndez

import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Load the dataset from a CSV file into a DataFrame.
df = pd.read_csv('paleteria.csv')
print('Dataset original:')
print(df)

# Shuffle the instances of the dataset to ensure randomness and avoid any bias related to the data order.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print('\nDataset shuffled:')
print(df)

# Separate the feature variables (predictors) from the target variable and display them.
X = df.drop(['y'], axis=1)
print('\nFeatures:')
print(X)

# Isolate the target variable 'y' and print it. This variable is what the model will predict.
y = df['y']
print('\nLabels:')
print(y)

# Split the dataset into a training subset (90%) and a testing subset (10%).
# This helps in validating the model on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('\nTraining set:')
print(X_train)
print(y_train)
print('\nTest set:')
print(X_test)
print(y_test)
