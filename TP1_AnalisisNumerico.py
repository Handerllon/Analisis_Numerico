# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 07:54:39 2018

@author: Manuel
"""


#Imports
from scipy.optimize import brentq
#import timeit Para calcular tiempo de corrida---VER SI USAR
import numpy as np
import matplotlib.pylab as plt

#Funciones f1, f2, f3 y derivadas
def f1(x): 
    return x**2-2   
def df1(x): 
    return 2*x
def ddf1(x): 
    return 2

def f2(x): 
    return x**5 - (6.6)*x**4 + (5.12)*x**3 + (21.312)*x**2 - (38.016*x) + 17.28
def df2(x): 
    return (5)*x**4 - (26.4)*x**3 + (15.36)*x**2 + (42.624*x) - 38.016
def ddf2(x): 
    return (20)*x**3 - (79.2)*x**2 + (30.72*x) + 42.624

def f3(x): 
    return (x - 1.5)*(np.exp((-4*(x - 1.5)**2)))
def df3(x): 
    return ((-8*x + 12)*(x - 1.5))*np.exp((-4*(x - 1.5)**2)) + np.exp((-4*(x - 1.5)**2))
def ddf3(x): 
    return (-24*x + (x-1.5)*(8*x - 12)**2 + 36)*np.exp((-4*(x - 1.5)**2))

#Funciones busqueda de raices
def bisec(f, a, b, a_tol, n_max):
    """
    Devolver (x0, delta), raiz y cota de error por metodo de la biseccion
    Datos deben cumplir f(a)*f(b) > 0
    """
    x = a+(b-a)/2    #mejor que (a+b)/2 segun Burden
    delta = (b-a)/2
    
    print('{0:^4} {1:^17} {2:^17} {3:^17}'.format('i', 'x', 'a', 'b'))
    print('{0:4} {1: .14f} {2: .14f} {3: .14f}'.format(0, x, a, b))
    
    for i in range(n_max):
        if f(a) * f(x) > 0:
            a = x
        else:
            b = x
        x_old = x
        x = a+(b-a)/2 #(a+b)/2
        delta = np.abs(x - x_old)
        
        print('{0:4} {1: .14f} {2: .14f} {3: .14f}'.format(i+1, x, a, b))
        
        if delta <= a_tol: #Hubo convergencia
            print('Hubo convergencia, n_iter = ' + str(i+1))
            return x, delta, i+1
    
    #Si todavia no salio es que no hubo convergencia:
    raise ValueError('No hubo convergencia')
    return x, delta, i+1

def secante(f, x0, x1, a_tol, n_max):
    """
    Devolver (x, delta), raiz y cota de error por metodo de la secante
    """
    delta = 0

    print('{0:^4} {1:^17} {2:^17} {3:^17}'.format('i', 'x', 'x_-1', 'delta'))
    print('{0:4} {1: .14f} {2: .14f} {3: .14f}'.format(0, x1, x0, delta))

    for i in range(n_max):
        x = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
        delta = np.abs(x - x1)
        x0 = x1
        x1 = x
        
        print('{0:4} {1: .14f} {2: .14f} {3: .14f}'.format(i+1, x1, x0, delta))
        
        #Chequear convergencia
        if delta <= a_tol: #Hubo convergencia
            print('Hubo convergencia, n_iter = ' + str(i+1))
            return x, delta, i+1

    #Si todavia no salio es que no hubo convergencia:
    #raise ValueError('No hubo convergencia')
    print ('No hubo convergencia')
    return x, delta, i+1

def newtonRaphson(f, df, x0, x1, toleranciaError, maximoIteraciones):
    """
    Devolver (x, delta), raiz y cota de error por metodo de la secante
    """
    delta = 0
    
    x = 1
    
    print('{0:^4} {1:^17} {2:^17}'.format('i', 'x','delta'))
    print('{0:4} {1: .14f} {2: .14f}'.format(0, x, delta))
    
    for i in range(maximoIteraciones):
        
        xSiguiente = x - (f(x)/df(x))
        delta = np.abs(xSiguiente - x)
                
        print('{0:4} {1: .14f} {2: .14f}'.format(i+1, xSiguiente, delta))
        
        x = xSiguiente        
        
        #Chequear convergencia
        if delta <= toleranciaError: #Hubo convergencia
            print('Hubo convergencia, n_iter = ' + str(i+1))
            return x, delta, i+1
    
    #Si todavia no salio es que no hubo convergencia:
    #raise ValueError('No hubo convergencia')
    print ('No hubo convergencia')
    return x, delta, i+1

def newtonRaphsonModificado(f, df, ddf, x0, x1, toleranciaError, maximoIteraciones):
    """
    Devolver (x, delta), raiz y cota de error por metodo de la secante
    """
    delta = 0
    
    x = 1
    
    print('{0:^4} {1:^17} {2:^17}'.format('i', 'x','delta'))
    print('{0:4} {1: .14f} {2: .14f}'.format(0, x, delta))
    
    for i in range(maximoIteraciones):
        
        xSiguiente = x - ((f(x)*df(x))/(df(x)**2-(f(x)*ddf(x))))
        delta = np.abs(xSiguiente - x)
                
        print('{0:4} {1: .14f} {2: .14f}'.format(i+1, xSiguiente, delta))
        
        x = xSiguiente        
        
        #Chequear convergencia
        if delta <= toleranciaError: #Hubo convergencia
            print('Hubo convergencia, n_iter = ' + str(i+1))
            return x, delta, i+1
    
    #Si todavia no salio es que no hubo convergencia:
    #raise ValueError('No hubo convergencia')
    print ('No hubo convergencia')
    return x, delta, i+1

def imprimirDatos (nombreFuncion,funcion, dFuncion, ddFuncion, xMin, xMax, error1, error2, maximasIteraciones):
    print('----------------')
    print('Metodo biseccion')
    print('----------------')
    print('')
    print('Funcion ' + nombreFuncion +', a_tol = '+str(error1))
    r, delta, n_iter = bisec(funcion, xMin, xMax, error1, maximasIteraciones)
    print('raiz = ' +str(r))
    print('delta= ' +str(delta))
    print('n_ite= ' +str(n_iter))
    print('')
    print('Funcion ' + nombreFuncion +', a_tol = '+str(error2))
    r, delta, n_iter = bisec(funcion, xMin, xMax, error2, maximasIteraciones)
    print('raiz = ' +str(r))
    print('delta= ' +str(delta))
    print('n_ite= ' +str(n_iter))
    print('')
    
    print('----------------')
    print('Metodo secante')
    print('----------------')
    print('')
    print('Funcion ' + nombreFuncion +', a_tol = '+str(error1))
    r, delta, n_iter = secante(funcion, xMin, xMax, error1, maximasIteraciones)
    print('raiz = ' +str(r))
    print('delta= ' +str(delta))
    print('n_ite= ' +str(n_iter))
    print('')
    print('Funcion ' + nombreFuncion +', a_tol = '+str(error2))
    r, delta, n_iter = secante(funcion, xMin, xMax, error2, maximasIteraciones)
    print('raiz = ' +str(r))
    print('delta= ' +str(delta))
    print('n_ite= ' +str(n_iter))
    print('')
    
    print('----------------')
    print('Metodo Newton-Raphson')
    print('----------------')
    print('')
    print('Funcion ' + nombreFuncion +', a_tol = '+str(error1))
    r, delta, n_iter = newtonRaphson(funcion, dFuncion, xMin, xMax, error1, maximasIteraciones)
    print('raiz = ' +str(r))
    print('delta= ' +str(delta))
    print('n_ite= ' +str(n_iter))
    print('')
    print('Funcion ' + nombreFuncion +', a_tol = '+str(error2))
    r, delta, n_iter = newtonRaphson(funcion, dFuncion, xMin, xMax, error2, maximasIteraciones)
    print('raiz = ' +str(r))
    print('delta= ' +str(delta))
    print('n_ite= ' +str(n_iter))
    print('')
    
    print('----------------')
    print('Metodo Newton-Raphson para raíces multiples')
    print('----------------')
    print('')
    print('Funcion ' + nombreFuncion +', a_tol = '+str(error1))
    r, delta, n_iter = newtonRaphsonModificado(funcion, dFuncion, ddFuncion, xMin, xMax, error1, maximasIteraciones)
    print('raiz = ' +str(r))
    print('delta= ' +str(delta))
    print('n_ite= ' +str(n_iter))
    print('')
    print('Funcion ' + nombreFuncion +', a_tol = '+str(error2))
    r, delta, n_iter = newtonRaphsonModificado(funcion, dFuncion, ddFuncion, xMin, xMax, error2, maximasIteraciones)
    print('raiz = ' +str(r))
    print('delta= ' +str(delta))
    print('n_ite= ' +str(n_iter))
    print('')
    
    print('----------------')
    print('Metodo brent')
    print('----------------')
    print('')
    print('Funcion ' + nombreFuncion +', a_tol por defecto para la funcion')
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
    r, results = brentq(funcion, xMin, xMax, full_output=True)
    print('raiz = ' +str(r))
    print('Resultados: ')
    print(results)
    
    #Grafica de las funciones
    #Ver https://matplotlib.org
    xx = np.linspace(xMin, xMax, 256+1)
    yy = funcion(xx)
    nombre = nombreFuncion
    plt.figure(figsize=(10,7))
    plt.plot(xx, yy, lw=2)
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel(nombre +'(x)')
    plt.title('Funcion '+ nombre)
    plt.grid(True)
    plt.savefig(nombre + '.png')
    plt.show()

#Intervalo para buscar raiz
xMin = 0.0
xMax = 2.0

#Parametros para el algoritmo
error1 = 1e-5
error2 = 1e-13
maximasIteraciones = 100

#imprimirDatos("f1", f1, df1, ddf1, xMin, xMax, error1, error2, maximasIteraciones)
#imprimirDatos("f2", f2, df2, ddf2, xMin, xMax, error1, error2, maximasIteraciones)
imprimirDatos("f3", f3, df3, ddf3, xMin, xMax, error1, error2, maximasIteraciones)


        