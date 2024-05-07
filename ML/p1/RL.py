import numpy as np

# Ingreso manual de los arreglos x e y
x = np.array([3,7,11,15,18,27,29,30,30,31,31,32,33,33,34,36,36,36,37,38,39,39,39,40,41,42,42,43,44,45])
y = np.array([5,11,21,16,16,28,27,25,35,30,40,32,34,32,34,37,38,34,36,38,37,36,45,39,41,40,44,37,44,46])


# Calculamos los valores de b0 y b1
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
b1 = numerator / denominator
b0 = y_mean - b1 * x_mean

# Impresión de los valores de b0 y b1
print(f"b0: {b0}")
print(f"b1: {b1}")

# Cálculo de la recta de regresión estimada
x_min = np.min(x)
x_max = np.max(x)
x_line = np.linspace(x_min, x_max, 100)
y_line = b0 + b1*x_line

# Impresión de la recta de regresión estimada
print("Recta de regresión estimada:")
print(f"y = {b0} + {b1}*x")

# Calculamos los datos estimados
y_estimated = b0 + b1 * x

# Calculamos la suma de las diferencias entre los datos de las muestras de prueba y los datos estimados
y_test = np.array([46,49,51])
y_predicted = b0 + b1 * np.array([46,47,50])
sum_diff = np.sum(y_test - y_predicted)

# Imprimimos la suma de las diferencias
print("La suma de las diferencias entre los datos de las muestras de prueba y los datos estimados es:", sum_diff)
