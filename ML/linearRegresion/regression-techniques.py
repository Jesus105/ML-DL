# Advanced Regression Techniques Implementation
# This script includes implementations for various regression methods including Ordinary Least Squares (OLS),
# Batch Gradient Descent (BGD) both univariate and multivariate, and Stochastic Gradient Descent (SGD).
# Users can select the desired method via a menu-driven interface.
#Author: Jesús Méndez

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ols_regression(file_name):
    # Load data and calculate regression parameters using Ordinary Least Squares method.
    df = pd.read_csv(file_name)
    x = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_mean = np.mean(x*y)
    xx_mean = np.mean(x*x)
    slope = (x_mean * y_mean - xy_mean) / (x_mean**2 - xx_mean)
    intercept = y_mean - slope * x_mean
    eq = f'y = {intercept} + {slope}*x'
    plt.scatter(x, y, color='blue')  # Plot data points
    plt.plot(x, slope * x + intercept, color='red')  # Plot regression line
    plt.title('OLS Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return eq

def BGD_Monovariable(w=0, a=0.01, iter=5):
    # Perform Batch Gradient Descent for a single variable linear regression.
    dataframe = pd.read_csv("mono.csv")
    x = dataframe["x"].values
    y = dataframe["y"].values
    wHistory = []
    for i in range(iter):
        sum = 0
        for j in range(len(x)):
            sum += (w*x[j]-y[j])*x[j]
        w -= a * 2 * sum
        wHistory.append(w)
    for i in range(iter):
        print(f'Iter {i}: {wHistory[i]}')
    y_pred = np.dot(x, w)  # Predict y values
    print(f"Regression Equation: Y = {w}*x")

def BGD_Multivariable(iter=5):
    # Perform Batch Gradient Descent for multivariable linear regression.
    dataframe = pd.read_csv("multi.csv")
    x = dataframe[['x0', 'x1', 'x2']].values
    y = dataframe['y'].values
    w = np.zeros(3)  # Initialize weights as zeros
    a = 0.01  # Learning rate
    wHistory = [[], [], []]
    for i in range(len(x)):
        for j in range(iter):
            sum = 0
            for k in range(len(x)):
                sum += (w[i]*x[i][k]-y[k])*x[i][k]
            w[i] -= a * 2 * sum
            wHistory[i].append(w[i])
    for i in range(iter):
        print(f'Iter {i}: W0: {round(wHistory[0][i], 2)}, W1: {round(wHistory[1][i], 2)}, W2: {round(wHistory[2][i], 2)}')
    print(f"Regression Equation: Y = {w[0]}*x0 + {w[1]}*x1 + {w[2]}*x2")
    
def SGD_MonoVariable():
    print("\n------------------------------------------------------------")
    print("\033[34mGradiente descendente estocástico (SGD) Monovariable\033[0m")
    df=pd.read_csv("mono.csv", sep=",")

    x=df["x"].values
    y=df["y"].values

    w=0
    a=0.01

    nIter=int(input('Cuantas iteraciones quieres: '))
    valoresJ=[]

    for i in range(nIter):
        print("Valor de J[",i,"] : ")
        valorJ=int(input())
        valoresJ.append(valorJ)

    for j in valoresJ:
        gradient = -2*x[j]*(y[j]-x[j]*w)
        w = w-(a*gradient)
        print(f'Con j = {j} w = {w}')
    
    print(f"Ecuación de Y =", w,"*x")




def SGD_MultiVariable():
    print("------------------------------------------------------------")
    print("\033[34mGradiente descendente estocástico (SGD) Multivariable\033[0m")
    df=pd.read_csv("multi.csv", sep=",")

    x=df[['x0', 'x1', 'x2']].values
    y=df["y"].values

    w=np.zeros(x.shape[0])
    a=0.01
    k=len(x)

    nIter=int(input('Cuantas iteraciones quieres: '))
    valoresJ=[]

    for i in range(nIter):
        print("Valor de J[",i,"] :",end='')
        valorJ=int(input())
        valoresJ.append(valorJ)

    for index in range(nIter):

        j = valoresJ[index]

        for i in range(len(x)):
            w[i] = np.round(w[i] - a * 2 * (np.dot((np.dot(w[i], x[i][j]) - y[j]), x[i][j])), 2)

        print(f'Iter {index} con j = {valoresJ[i]} w = {w}')
        
    
    print(f"Ecuación de Y =",w[0],"* x0 +",w[1],"* x1 + ",w[2]," * x2")

# Main program with a menu to choose the regression method
if __name__ == '__main__':
    while True:
        print("[1] OLS")
        print("[2] BGD Monovariable")
        print("[3] BGD Multivariable")
        print("[4] SGD Monovariable")
        print("[5] SGD Multivariable")
        print("[0] Exit")
        try:
            option = int(input("Select an option: "))
        except:
            print('Invalid option')
        if option == 1:
            print("------------------------------------------------------------")
            print("OLS Linear Regression")
            print(ols_regression("ols.csv"))
        elif option == 2:
            print("------------------------------------------------------------")
            print("BGD Single Variable")
            try:
                iter = int(input("Enter the number of iterations: "))
            except:
                print('Invalid option')
            BGD_Monovariable(iter=iter)
        elif option == 3:
            print("------------------------------------------------------------")
            print("BGD Multivariable")
            try:
                iter = int(input("Enter the number of iterations: "))
            except:
                print('Invalid option')
            BGD_Multivariable(iter=iter)
        elif option == 4:
            SGD_MonoVariable()
        elif option == 5:
            SGD_MultiVariable()
        elif option == 0:
            print("Goodbye :)")
            break
