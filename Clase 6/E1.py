# Regresion Logistica --> Sigmoid -->  Salida C(0,1)

import numpy as np
import matplotlib.pyplot as plt
import time


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('X_1', np.float),
                     ('X_2', np.float),
                     ('A', np.int)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]),
                         np.int(line.split(',')[2]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):
        X = self.dataset[['X_1', 'X_2']]
        y = self.dataset['A']

        permuted_idxs = np.random.permutation(len(X))

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test


class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __init__(self):
        Metric.__init__(self)

    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n


def mini_batch_logistic_regression(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 10
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            exponent = np.sum(np.transpose(W) * batch_X, axis=1)
            prediction = 1/(1 + np.exp(-exponent))
            error = prediction.reshape(-1, 1) - batch_y.reshape(-1, 1)

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = 1/b * grad_sum
            gradient = np.transpose(grad_mul).reshape(-1, 1)

            W = W - (lr * gradient)

    return W



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




def predict(X, W):
    wx1 = np.matmul(X, W)
    wx = np.concatenate(wx1, axis=0)
    sigmoid = 1 / (1 + np.exp(-wx))
    # sigmoid = [1 if x >= 0.5 else 0 for x in sigmoid]
    sigmoid = np.where(sigmoid >= 0.5, 1, 0)
    return sigmoid


if __name__ == '__main__':

    # Creamos el objeto dataset para la carga del archivo
    dataset = Data('clase_6_dataset.csv')

    # Dividimos el dataset
    X_train, X_test, y_train, y_test = dataset.split(0.8)

    # Expandimos los datos de X para agregar la columna de 1
    X_train_expanded = np.vstack((X_train['X_1'], X_train['X_2'], np.ones(len(X_train)))).T
    X_test_expanded = np.vstack((X_test['X_1'], X_test['X_2'], np.ones(len(X_test)))).T

    # Establecemos los parametros para learning rate, epochs y estimamos W con Mini Bach
    lr = 0.001
    epochs = 50000
    print('Training....')
    start = time.time()

    W = mini_batch_logistic_regression(X_train_expanded, y_train.reshape(-1, 1), lr=lr, amt_epochs=epochs)

    time = time.time() - start
    print('W: {}\nTime [s]: {}'.format(W.reshape(-1), time))

    # Calculamos la y hat
    # y_hat = w0 * x1 + w1 * x2 + w3
    # 0 = w0 * x1 + w1 * x2 + w3
    # X2(y) = ( -w0 *x1(x) - w3) / w1
    y_hat = predict(X_test_expanded, W)
    print("Val. Real:")
    print(y_test)
    print("Prediccion:")
    print(y_hat)

    # Plots del modelo Regresion Logistica
    x_regression = np.linspace(30, 100, 70)
    y_regression = (-x_regression * W[0] - W[2]) / W[1]

    # Filtramos los resultados en valores discretos 1 y 0
    zeros = y_train < 0.5
    ones = y_train >= 0.5

    # Obtengo dos datasets separados para los 1 y 0
    X_train_zeros = X_train[zeros]
    y_train_zeros = y_train[zeros]

    X_train_ones = X_train[ones]
    y_train_ones = y_train[ones]

    plt.scatter(X_train_zeros['X_1'], X_train_zeros['X_2'])#, marker='*')
    plt.scatter(X_train_ones['X_1'], X_train_ones['X_2'])#, marker='+')
    plt.plot(x_regression, y_regression, c='r')
    plt.show()

