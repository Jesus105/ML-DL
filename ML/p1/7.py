import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Leer el archivo csv y convertirlo en un DataFrame
df = pd.read_csv('paleteria.csv')
print('Dataset original:')
print(df)

# Revolver las instancias del dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print('\nDataset revuelto:')
print(df)

# Separa las características del DataFrame y las imprime
X = df.drop(['y'], axis=1)
print('\nCaracterísticas:')
print(X)

# Separa las etiquetas del DataFrame y las imprime
y = df['y']
print('\nEtiquetas:')
print(y)

# Separa el dataset en un subconjunto de entrenamiento (90%) y un subconjunto de prueba (10%) y los imprime como salida
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('\nConjunto de entrenamiento:')
print(X_train)
print(y_train)
print('\nConjunto de prueba:')
print(X_test)
print(y_test)
