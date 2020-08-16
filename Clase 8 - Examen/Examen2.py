import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold



class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('X', np.float),
                     ('Y', np.float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(', ')[0]), float(line.split(',')[1]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):
        X = self.dataset['X']
        y = self.dataset['Y']

        permuted_idxs = np.random.permutation(len(X))

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test


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


class LinearRegression(BaseModel):

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


def mini_batch_gradient_descent(X_train, y_train, lr=0.0001, amt_epochs=100):

    # Divido el dataset en 4/5 para el train y 1/5 para validacion

    permuted_idxs = np.random.permutation(X_train.shape[0])
    split_4_5 = int(0.8 * X_train.shape[0])
    train_idxs = permuted_idxs[0:split_4_5]
    val_idxs = permuted_idxs[split_4_5: X_train.shape[0]]
    X_train_mb = X_train[train_idxs]
    X_validation = X_train[val_idxs]
    y_train_mb = y_train[train_idxs]
    y_validation = y_train[val_idxs]


    b = 16

    n = X_train_mb.shape[0]
    m = X_train_mb.shape[1]
    mse = MSE()
    error_validation = []
    error_train = []
    # Paso 1 : Inicializamos W con un valor aleatorio  W  --> mx1  m = 1 si esf rl  m = 2 en rl+b

    W = np.random.randn(m).reshape(m, 1)
    for i in range(amt_epochs):
        # tomamos las muestras de forma aleatoria del dataset
        idx = np.random.permutation(X_train_mb.shape[0])
        X_train_mb = X_train_mb[idx]
        y_train_mb = y_train_mb[idx]

        error_t = 0
        # el tam del bach es n/b
        batch_size = int(len(X_train_mb) / b)
        # Recorremos el dataset con saltos del tam del bach
        for i in range(0, len(X_train_mb), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train_mb) else len(X_train_mb)
            batch_X = X_train_mb[i: end]
            batch_y = y_train_mb[i: end]

            prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1
            error_t += error.mean()
            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2/1000 * grad_sum  # 1xm  Deberia dividir por b, pero el grad. se va a valores infinitos
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)
        #print(W)
        error_t_mean = error_t / b
        error_train.append(error_t_mean)
        y_hat_train = np.matmul(X_validation, W)
        error_val = mse(y_hat_train,y_validation)
        error_validation.append(error_val)


    return W, error_validation, error_train


if __name__ == '__main__':

    # Levanto el dataser en un arreglo estructurado
    dataset = Data('clase_8_dataset.csv')

    # Gafico el dataset
    # plt.scatter(dataset.dataset['X'], dataset.dataset['Y'],label='dataset')
    # plt.legend()
    # plt.show()

    # Partimos el dataset en 80/20
    X_train, X_test, y_train, y_test = dataset.split(0.8)

    mse = MSE()
    lista_modelos = []

    # Elegir polinomio que logra mejor fit
    for n in range(0,4):
        # Armamos el polinomio de grado n
        X_train_l = [np.power(X_train, i) for i in range(0, n + 1)]
        X_train_pol = np.vstack((X_train_l)).T
        kf = KFold(n_splits=5, shuffle=True)
        mejor_modelo = []
        k=1
        for train_idx, test_idx in kf.split(X_train_pol):

            new_X_train, new_X_test = X_train_pol[train_idx], X_train_pol[test_idx]
            new_y_train, new_y_test = y_train[train_idx], y_train[test_idx]
            # Creamos el modelo y lo entrenamos
            linRegression = LinearRegression()
            linRegression.fit(new_X_train, new_y_train)

            y_hat = linRegression.predict(new_X_test)

            ms_error = mse(new_y_test, y_hat)
            if (n==1):
                print("Pol. n = 1 - MSE para k {} : {:.2f} ".format(k,ms_error))
                k+=1

            # Criterio: Evaluo si es error cuadratico medio es menor al modelo_k anterior y lo guardo como mejor modelo
            # Realizo el mismo criterio para los polinomios de n grados guardando el modelo y n para recuperarlo luego
            if not mejor_modelo:
                mejor_modelo.append(ms_error)
                mejor_modelo.append(linRegression)
                mejor_modelo.append(n)
            else:
                if ms_error < mejor_modelo[0]:
                    mejor_modelo[0] = ms_error
                    mejor_modelo[1] = linRegression
                    mejor_modelo[2] = n

        # Almaceno el mejor modelo del polinomio evaluado
        if(n==1):
            print("Mejor fit del modelo lineal (Pol. n=1) : mse {:.2f}".format(mejor_modelo[0]))
        lista_modelos.append(mejor_modelo)

    mejor_modelo = lista_modelos[0]
    for modelo in lista_modelos[1:]:
        if modelo[0] < mejor_modelo[0]:
            mejor_modelo = modelo

    # Recupero el modelo con menor mse y creo el polinomio de grado con los datos del test para evaluar el error
    linRegression = mejor_modelo[1]
    n = mejor_modelo[2]

    # Creo el polinomio de test de grado n con los datos del test
    X_test_l = [np.power(X_test, i) for i in range(0, n + 1)]
    X_test_pol = np.vstack((X_test_l)).T

    y_hat = linRegression.predict(X_test_pol)
    ms_error = mse(y_test, y_hat)
    print("El polinomio que mejor aproxima es de grado n = {} con MSE = {:.2f}".format(mejor_modelo[2],ms_error))
    plt.scatter(dataset.dataset['X'], dataset.dataset['Y'], label='dataset')
    plt.scatter(X_test, y_hat, label=f'pol. Grado n={n}')
    #plt.legend()
    #plt.show()

    # Creo el modelo de grado 2 para calcular el gradiente de mini batch.
    # Con n =2 se vuelve bastante inestable, en ocaciones se logra una aproximacion adecuada, y en otras el modelo
    # tiende a irse a valores altos
    n = 2 #
    X_train_mb = [np.power(X_train, i) for i in range(0, n + 1)]
    X_train_mb_pol = np.vstack((X_train_mb)).T
    X_test_l = [np.power(X_test, i) for i in range(0, n + 1)]
    X_test_pol = np.vstack((X_test_l)).T

    print("\nOptimizacion con Gradiente descendente Mini Batch n = 2")
    lr = 0.00000001
    epochs = 1000
    # Divido en dataset Train en 80 / 20  para validar el mini batch
    W, error_val, error_train = mini_batch_gradient_descent(X_train_mb_pol, y_train.reshape(-1, 1), lr=lr, amt_epochs=epochs)
    y_hat_mb = np.matmul(X_test_pol, W)

    plt.scatter(X_test, y_hat_mb, label=f'Mini Batch con pol grado n={n}')
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.subplot(1, 1)
    # error_val.pop(0) # quito el primer elemento para mejorar la apreciacion del grafico
    # error_train.pop(0)
    x_error_val = np.linspace(1, epochs , epochs)
    # plt.scatter(x_error_val,error_val,label=f'Error de validacion por cada epoch')
    #
    # plt.subplot(2, 1)
    x_error_train = np.linspace(1, epochs, epochs)
    # plt.scatter(x_error_train, error_train, label=f'Error de train por cada epoch')
    # plt.legend()
    # plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('Mini Batch con Pol. n = 2')
    axs[0].plot(x_error_val, error_val)
    axs[0].set_title('Error de validacion por cada epoch')
    axs[1].plot(x_error_train, error_train,'tab:red')
    axs[1].set_title('Error de train por cada epoch')
    plt.show()