# Carga un dataset y genera la regresion lineal con polinomios
import numpy as np
from matplotlib import pyplot as plt



class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)



    def _build_dataset(self, path):
        # levanta el dataset en un array estructurado

        structure = [('income',np.float),
                     ('happiness',np.float)]

        with open(path, encoding="utf8" ) as data_csv:
            # Creamos un generador, similar a una lista pero no se
            # carga completo en memoria (puntero que levanta dato a dato)
            data_gen = ((float(line.split(',')[1]), float(line.split(',')[2]))
                        for i, line in enumerate(data_csv) if i !=0)
            # pasamos al iterador el generator y la estructura
            embeddings = np.fromiter(data_gen, structure)

        return embeddings


    def split(self, percentage):
        # divide el dataset segun el porcentaje dado

        x = self.dataset['income']
        y = self.dataset['happiness']

        # obtenemos los indices de forma aleatoria de x.shape[0]: cant. filas
        permuted_index_x = np.random.permutation(x.shape[0])
        # dividimos el indice en los porcentajes dados
        train_index = permuted_index_x[0:int(percentage * x.shape[0])]
        test_index = permuted_index_x[int(percentage * x.shape[0]) : x.shape[0]]
        # obtenemos los dataset divididos
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        return x_train, x_test, y_train, y_test

class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        # train model
        return NotImplemented

    def predict(self, X):
        # retorna y hat
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, X, Y):
        # Calcular los W y guadarlos en el modelo
        W = Y.mean()
        self.model = W

    def predict(self, X):
        # usar el modelo (self.model) y predecir
        # y hat a partir de X e W
        return np.ones(len(X)) * self.model


class LinearRegresion(BaseModel):

    def fit(self, X, Y):
        # Calcular los W y guadarlos en el modelo
        if X.ndim == 1:
            W = X.T.dot(Y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        self.model = W

    def predict(self, X):
        # usar el modelo (self.model) y predecir
        # y_hat a partir de X e W

        if X.ndim == 1:
            return self.model * X
        else:
            return np.matmul(X, self.model)


class Metric(object):
    def __call__(self, target, prediction):
        # target --> Y
        # prediction --> Y hat
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        # Implementar el error cuadratico medio MSE
        return np.square(target - prediction).mean()


if __name__ == '__main__':

    dataset = Data('income.data.csv')

    X_train, X_test, y_train, y_test = dataset.split(0.8)

    lr = LinearRegresion()
    #train_mse = []
    vec_test_mse = []

    # Calculo el error cuadratico medio por cada n grado de polinomio para evaluar en n optimo
    for n in range(0, 12):
        # Creamos una lista con los elementos de X evaluados en cada exponente del polinomio hasta n
        X_train_l = [np.power(X_train, i) for i in range(0, n + 1)]
        X_test_l = [np.power(X_test, i) for i in range(0, n + 1)]

        # Agregamos al X  el polinomio transpuesto  X_train_l = [ 1 ....1] , [ ... ]^2 ..
        X_train_exp = np.vstack((X_train_l)).T
        X_test_exp = np.vstack((X_test_l)).T


        # Entreno la regresion lineal
        lr.fit(X_train_exp, y_train)

        # creamos un objeto de la clase mean square error
        mse = MSE()

        # calculamos y_hat para el polinomio n
        y_hat_train = lr.predict(X_train_exp)
        y_hat_test = lr.predict(X_test_exp)

        #mse_train = mse(y_train, y_hat_train)
        mse_test = mse(y_test, y_hat_test)

        #train_mse.append(mse_train)
        vec_test_mse.append(mse_test)


    #plt.plot(val_mse, color='b', label='mse []')
    plt.plot(vec_test_mse, color='g', label='mse y_test')
    plt.legend()
    plt.show()

