import pandas as pd
from sklearn.model_selection import train_test_split

# Leer el archivo csv y convertirlo en un DataFrame
df = pd.read_csv('paleteria.csv')
print('Dataset:')
print(df)

# Separa las características del DataFrame y las imprime
X = df.drop(['y'], axis=1)
print('\nCaracterísticas:')
print(X)

# Separa las etiquetas del DataFrame y las imprime
y = df['y']
print('\nEtiquetas:')
print(y)

# Separa el dataset en un subconjunto de entrenamiento (50%) y un subconjunto de prueba (50%) y los imprime como salida
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print('\nConjunto de entrenamiento:')
print(X_train)
print(y_train)
print('\nConjunto de prueba:')
print(X_test)
print(y_test)
