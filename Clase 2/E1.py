# E1/C2 : Numpy Broadcasting con matrices
import numpy as np


X = np.array([
     [1,2,3],
     [4,5,6],
     [7,8,9]
])

C = np.array([
    [1,0,0],
    [0,1,1]
])

Cexpand = C[:, None]    # Agrega una dimension a la matriz C quedando ahora de 2*1*3 para poder realizar el broadcasting

# Calculamos la distancia  haciendo la resta entre cada componente del vector fila de X por cada uno de los comp.
# fila de C y Suma por fila (axis 2)

distancias = np.sqrt(np.sum((Cexpand-X)**2,axis=2))

print(X)
print(Cexpand)

print(distancias)
