import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def k_folds(X_train, y_train, k=5):
    l_regression = LinearRegresion()
    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list = []
    for i in range(0, len(X_train), chunk_size):
        # divide el data_train en k bloques
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        # concateno los bloques
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train)
        prediction = l_regression.predict(new_X_valid)
        mse_list.append(error(new_y_valid, prediction))

    mean_MSE = np.mean(mse_list)

    return mean_MSE


if __name__ == '__main__':
    # senoidal con desvio 0.4
    sample_size = 1000
    X = np.linspace(0, 2 * np.pi, sample_size)
    y = np.sin(X) + np.random.normal(loc=0, scale=0.40, size=sample_size)

    # con train test split obtenemos la division del dataset de forma aleatoria
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(0.2))

    # Creamos el objeto de mean square error
    mse = MSE()
    best_models_n = 0
    lr = LinearRegresion()
    # hacemos un fit de polinomios hasta 10
    for n in range(12):
        # Creamos una lista con los elementos de X_train evaluados en cada exponente del polinomio hasta n
        X_train_l = [np.power(X_train, i) for i in range(0, n + 1)]
        # Agregamos al X  el polinomio transpuesto  X_train_l = [ 1 ....1] , [ ... ]^2 ..
        X_train_exp = np.vstack((X_train_l)).T

        # Utilizo kf para dividir mi train dataset y evaluar el mejor polinomio

        mean_MSE = k_folds(X_train_exp, y_train, k=5)
        #print(mean_MSE)
        if n == 0 : best_model = mean_MSE

        if best_model > mean_MSE:
            # No hay mejor modelo almacenado
            best_model = mean_MSE
            best_model_n = n

        # Almaceno el mejor modelo del polinomio evaluado
        print("Polinomio grado:", n, 'best MSE', mean_MSE)


    print('El modelo de mejor polinomio es n=', best_model_n)

    n = best_model_n
    # Creo el polinomio de test de grado n donde el error es menor
    X_train_l = [np.power(X_train, i) for i in range(0, n + 1)]
    X_test_list = [np.power(X_test, i) for i in range(0, n + 1)]
    X_train_exp = np.vstack((X_train_l)).T
    X_test_exp = np.vstack((X_test_list)).T

    lr.fit(X_train_exp, y_train)
    y_hat = lr.predict(X_test_exp)
    mse_test = mse(y_test, y_hat)

    print('Test MSE:', mse_test)

    # Graficos
    plt.scatter(X, y, label='dataset')
    plt.scatter(X_test, y_hat, label=f'poly n={n}')
    plt.legend()
    plt.show()
