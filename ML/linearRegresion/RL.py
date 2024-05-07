# Simple Linear Regression Implementation
# This script demonstrates the implementation of a simple linear regression model.
# It calculates the linear relationship between two variables using the least squares method.
#Author: Jesús Méndez

import numpy as np

# Manually entered arrays x and y, representing independent and dependent variables respectively.
x = np.array([3, 7, 11, 15, 18, 27, 29, 30, 30, 31, 31, 32, 33, 33, 34, 36, 36, 36, 37, 38, 39, 39, 39, 40, 41, 42, 42, 43, 44, 45])
y = np.array([5, 11, 21, 16, 16, 28, 27, 25, 35, 30, 40, 32, 34, 32, 34, 37, 38, 34, 36, 38, 37, 36, 45, 39, 41, 40, 44, 37, 44, 46])

# Calculate the coefficients b0 and b1 using the formula for linear regression.
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
b1 = numerator / denominator
b0 = y_mean - b1 * x_mean

# Print the regression coefficients
print(f"b0 (Intercept): {b0}")
print(f"b1 (Slope): {b1}")

# Calculate the regression line from the min to max values of x for visualization.
x_min = np.min(x)
x_max = np.max(x)
x_line = np.linspace(x_min, x_max, 100)
y_line = b0 + b1 * x_line

# Print the equation of the estimated regression line
print("Estimated Regression Line:")
print(f"y = {b0} + {b1} * x")

# Calculate estimated y values based on the regression model
y_estimated = b0 + b1 * x

# Calculate and print the sum of differences between test data and predicted data
y_test = np.array([46, 49, 51])
y_predicted = b0 + b1 * np.array([46, 47, 50])
sum_diff = np.sum(y_test - y_predicted)

# Print the sum of differences between the test samples and the estimated data
print("Sum of differences between test data and estimated data:", sum_diff)
