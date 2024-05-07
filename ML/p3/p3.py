import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ols_regression(file_name):

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

    plt.scatter(x, y, color='blue')
    plt.plot(x, slope * x + intercept, color='red')
    plt.title('Regresión lineal OLS')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return eq

def BGD_Monovariable(w = 0, a = 0.01, iter = 5):
    dataframe = pd.read_csv("mono.csv")
    print(dataframe)

    x = dataframe["x"].values
    y = dataframe["y"].values
    
    wHistory = list()

    for i in range(iter):
        sum = 0
        
        for j in range(len(x)):
            sum += (w*x[j]-y[j])*x[j]

        w = w - (a*2*sum)
        wHistory.append(w)


    for i in range(iter):
        print(f'Iter {i}: {wHistory[i]}')    

    
    print(f"Ecuacion de Y = {w}",'x\n')

    # Predicción de valores de y
    y_pred = np.dot(x, w)

def BGD_Multivariable(iter = 5):
    dataframe = pd.read_csv("multi.csv")
    print(dataframe)

    x = dataframe[['x0', 'x1', 'x2']].values
    y = dataframe['y'].values

    w = np.zeros(3)  # w = [0, 0, 0]
    a = 0.01
    k = len(x)
    
    wHistory = [[],[],[]]

    for i in range(len(x)):
        for j in range(iter):
            sum = 0
            for k in range(len(x)):
                sum += (w[i]*x[i][k]-y[k])*x[i][k]
            w[i] = (w[i] - (a*2*sum))

            wHistory[i].append(w[i])
            
        
    for i in range(iter):
        print(f'Iter {i}: W0: {round(wHistory[0][i],2)}, W1: {round(wHistory[1][i],2)}, W2: {round(wHistory[2][i],2)}')


    print(f"Ecuacion de Y =", round(w[0],2), "+", round(w[1],2), "x1 +", round(w[2],2), "x2\n")


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


if __name__ == '__main__':
    while True:
        print("[1] OLS")
        print("[2] BGD Monovariable")
        print("[3] BGD Multivariable")
        print("[4] SGD Monovariable")
        print("[5] SGD Multivariable")
        print("[0] Salir")
        try:
            opcion = int(input("Seleccione una opcion: "))
        except:
            print('Opcion no valida')

        if opcion == 1:
            print("------------------------------------------------------------")
            print("\033[34mRegresión lineal medianteMínimos Cuadrados Ordinarios (OLS) \033[0m")
            print(ols_regression("ols.csv"))
        elif opcion == 2:
            print("------------------------------------------------------------")
            print("\033[34mGradiente de la función de perdida (BGD) Monovariable \033[0m")
            try:
                iter = int(input("Ingrese el numero de iteraciones: "))
            except:
                print('Opcion no valida')
            BGD_Monovariable(iter=iter)
        elif opcion == 3:
            print("------------------------------------------------------------")
            print("\033[34mGradiente de la función de perdida (BGD) Multivariable \033[0m")
            try:
                iter = int(input("Ingrese el numero de iteraciones: "))
            except:
                print('Opcion no valida')
            BGD_Multivariable(iter = iter)
        elif opcion == 4:
           SGD_MonoVariable()
        elif opcion == 5:
            SGD_MultiVariable()
        elif opcion == 0 :
            print("Hasta luego :)")
            break