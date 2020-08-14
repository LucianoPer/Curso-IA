# Calculo de un PSA para reducir dimension
# Calculo desde la libreria sklearn

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


x = np.array([
    [0.4 , 4800 , 5.5],
    [0.7 , 12104 , 5.2],
    [1, 12500 , 5.5],
    [1.5 , 7002 , 4.0]
])


def PCA_function_sk(dataset_X):
    pca = PCA(n_components=2)
    x_std = StandardScaler(with_std=False).fit_transform(dataset_X)
    return  pca.fit_transform(x_std)


def my_PCA_function(dataset_X):
    # calculamos el valor medio para centrar el dataset
    dataset_X_mean = dataset_X.mean(axis=0)
    X = dataset_X - dataset_X_mean
    # calculamos la covarianza de X transpuesta
    cov_X = np.cov(X.T)
    # obtenemos los autovalores "w" y autovectores "v"
    w,v = np.linalg.eig(cov_X)
    # obtenemos el indice ordenado de mayor a menor de los autovalores (::-1)
    index = w.argsort()[::-1]
    #ordenamos los autovalores y autovectores de mayor a menor
    w = w[index]
    v = v[index]
    # proyectamos la matriz centrada por los "d" autovalores y autovectores mas relevantes
    return np.matmul(X,v[:,:2]) # d=2



print(PCA_function_sk(x))
print(my_PCA_function(x))