# Validation Techniques in Machine Learning
# This script demonstrates three common validation methods: Cross-Validation (K-Fold), Leave-One-Out, and Bootstrap.
# These techniques are crucial for assessing the performance and robustness of machine learning models.
#Author: Jesús Méndez

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.utils import resample

# Load the dataset and create a DataFrame
dataset = pd.read_csv('metodosDeValidacion.csv', sep=',')
df = pd.DataFrame(dataset)

def validacionCruzada(X_train, y_train):
    # Perform K-Fold Cross-Validation
    print("Cross-Validation:\n")
    kf = KFold(n_splits=4)

    print("For x:")
    # Split the data into training and validation sets
    for train_index, val_index in kf.split(X_train):
        X_trainVC, X_val = [X_train[i] for i in train_index], [X_train[i] for i in val_index]
        print(X_trainVC, X_val)

    print("For y:")
    for train_index, val_index in kf.split(y_train):
        y_trainVC, y_val = [y_train[i] for i in train_index], [y_train[i] for i in val_index]
        print(y_trainVC, y_val)

def dejarUnoAfuera(X_train, y_train):
    # Perform Leave-One-Out Validation
    loo = LeaveOneOut()
    print("For x:")
    for train_index, val_index in loo.split(X_train):
        X_trainLOO, X_val = [X_train[i] for i in train_index], [X_train[i] for i in val_index]
        print(X_trainLOO, X_val)

    print("For y:")
    for train_index, val_index in loo.split(y_train):
        y_trainLOO, y_val = [y_train[i] for i in train_index], [y_train[i] for i in val_index]
        print(y_trainLOO, y_val)

def boostrap(X_train, y_train, n_iterations, m_training):
    # Perform Bootstrap validation
    for i in range(n_iterations):
        # Generate a training subset via sampling with replacement
        X_trainB, y_trainB = resample(X_train, y_train, n_samples=m_training)
        # Generate a validation set from elements not in the training subset
        X_val, y_val = [], []
        for j in range(len(X_train)):
            if X_train[j] not in X_trainB:
                X_val.append(X_train[j])
                y_val.append(y_train[j])
        print("X", X_trainB, X_val)
        print("Y ", y_trainB, y_val)

if __name__ == '__main__':
    # Split dataset into training and test subsets
    X = dataset['x'].values
    y = dataset['y'].values
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=0)

    print("CROSS-VALIDATION:\n")
    validacionCruzada(X_entrenamiento, y_entrenamiento)
    print("LEAVE ONE OUT:\n")
    dejarUnoAfuera(X_entrenamiento, y_entrenamiento)
    print("BOOTSTRAP:\n")
    boostrap(X_entrenamiento, y_entrenamiento, 4, 12)
