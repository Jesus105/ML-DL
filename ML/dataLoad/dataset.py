# Dataset Processing for Model Training and Testing
# This script focuses on processing a dataset for use in machine learning models, including loading,
# splitting features and labels, and dividing the data into training and testing subsets.
#Author: Jesús Méndez

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file named 'paleteria.csv' into a pandas DataFrame.
df = pd.read_csv('paleteria.csv')
print('Dataset:')
print(df)

# Extract the feature variables (independent variables) from the DataFrame and display them.
# The 'y' column, assumed to be the label, is dropped from the features.
X = df.drop(['y'], axis=1)
print('\nFeatures:')
print(X)

# Isolate the target variable (dependent variable) 'y' from the DataFrame and print it.
# This column represents the labels that the model will attempt to predict.
y = df['y']
print('\nLabels:')
print(y)

# Split the dataset into training and testing subsets with a 50-50 split.
# This provides a balanced approach to training and testing the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print('\nTraining set:')
print(X_train)
print(y_train)
print('\nTest set:')
print(X_test)
print(y_test)
