import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.utils import resample

dataset = pd.read_csv('metodosDeValidacion.csv', sep=',')
df = pd.DataFrame(dataset)

def validacionCruzada(X_train,y_train):
   print("Validación cruzada:\n")
   kf = KFold(n_splits=4)

   print("Para x")

   for train_index, val_index in kf.split(X_train):
    # Obtener los conjuntos de entrenamiento y validación para esta iteración
    X_trainVC, X_val = [X_train[i] for i in train_index], [X_train[i] for i in val_index]
    # Hacer lo que sea necesario con los conjuntos de entrenamiento y validación para esta iteración
    print(X_trainVC, X_val)

   print("Para y")

   for train_index, val_index in kf.split(X_train):
    y_trainVC, y_val = [y_train[i] for i in train_index], [y_train[i] for i in val_index]   
    # Hacer lo que sea necesario con los conjuntos de entrenamiento y validación para esta iteración
    print(y_trainVC,y_val)


def dejarUnoAfuera(X_train,y_train):
   loo = LeaveOneOut()
   print("Para x")
   for train_index, val_index in loo.split(X_train):
    # Obtener los conjuntos de entrenamiento y validación para esta iteración
    X_trainLOO, X_val = [X_train[i] for i in train_index], [X_train[i] for i in val_index]
    # Hacer lo que sea necesario con los conjuntos de entrenamiento y validación para esta iteración
    
    print(X_trainLOO, X_val)

   print("Para y")
   for train_index, val_index in loo.split(X_train):
    y_trainLOO, y_val = [y_train[i] for i in train_index], [y_train[i] for i in val_index]
    # Hacer lo que sea necesario con los conjuntos de entrenamiento y validación para esta iteración
    
    print(y_trainLOO, y_val)


def boostrap(X_train,y_train,n_iteraciones,m_entrenamiento):

   for i in range(n_iteraciones):
      # Generar un conjunto de entrenamiento
      X_trainB,y_trainB = resample(X_train,y_train,n_samples=m_entrenamiento)
      # Generar un conjunto de validación con los elementos que no se encuentran en el conjunto de entrenamiento
      X_val, y_val = [], []
      for j in range(len(X_train)):
         if X_train[j] not in X_trainB:
               X_val.append(X_train[j])
               y_val.append(y_train[j])
      
      print("X", X_trainB, X_val)
      print("Y ", y_trainB, y_val)


if __name__ == '__main__': 
    
    X = dataset['x'].values
    y = dataset['y'].values
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba =  train_test_split(X,y, test_size=0.4, shuffle=False, random_state=0)

    print("VALIDACIÓN CRUZADA:\n")
    validacionCruzada(X_entrenamiento,y_entrenamiento)
    print("LEAVE ONE OUT:\n")
    dejarUnoAfuera(X_entrenamiento,y_entrenamiento)
    print("BOOTSTRAP:\n")
    boostrap(X_entrenamiento,y_entrenamiento,4,12)