# carga un dataset los splitea y realiza la regresion lineal

import numpy as np
import matplotlib.pyplot as plt


class Data(object):

    # En el constructor tomamos el path del archivo dado como parametro
    def __init__(self, path):
        self.dataset = self._build_dataset(path)
        # _ metodo privado _, solo se accede dentro de la clase


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

    def fit(self, x, y):
        return NotImplemented

    def predict(self, x):
        return NotImplemented


class ConstantModel(BaseModel):
    # Modelo basico para contrastar resultados: Calculo de promedio
    def fit(self, x, y):
        w = y.mean()
        self.model = w

    def predict(self, x):
        return np.ones(len(x)) * self.model


class LinearRegression(BaseModel):

    def fit(self, x, y):
        # si el array es de una sola dimension
        if len(x.shape) == 1:
            w = x.T.dot(y) / x.T.dot(x)
        # si el array es de mas de una dimension
        else:
            w = np.linalg.inv(x.T.doy(y)).dot(x.T).dot(y)
        self.model = w

    def predict(self, x):
        return self.model * x


class LinearRegressionWithB(BaseModel):

    def fit(self, x, y):
        # expandimos una columna del dataset con unos
        x_expanded = np.vstack((x, np.ones(len(x)))).T
        w = np.linalg.inv(x_expanded.T.dot(x_expanded)).dot(x_expanded.T).dot(y)
        self.model = w

    def predict(self, x):
        x_expanded = np.vstack((x, np.ones(len(x)))).T
        return x_expanded.dot(self.model)


class Metric(object):

    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):

    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction)**2) / n




if __name__ == '__main__':
    # Creo un objeto de la clase Data y le paso la direccion donde esta el archivo a procesar
    dataset = Data('./income.data.csv')

    # spliteamos el dataset en 80/20 para entrenar y evaluar las metricas
    x_train, x_test, y_train, y_test = dataset.split(0.8)

    # Modelo Regresion lineal sin constante
    regresion_lineal = LinearRegression()
    # Creamos un objeto de LinearR y entrenamos el modelo con fit
    regresion_lineal.fit(x_train, y_train)
    # Obtenemos la prediccion de las salidas con el dataset test
    y_hat_rl = regresion_lineal.predict(x_test)

    # Modelo Regresion Lineas con constante
    regresion_lineal_with_b = LinearRegressionWithB()
    regresion_lineal_with_b.fit(x_train,y_train)
    y_hat_rlb = regresion_lineal_with_b.predict(x_test)

    # Modelo constante
    modelo_constante = ConstantModel()
    modelo_constante.fit(x_train,y_train)
    y_hat_cte = modelo_constante.predict(x_test)

    # Calculamos la metrica en este caso es el error cuadratico medio de cada pred.
    mse = MSE()
    RL_mse = mse(y_test, y_hat_rl)
    RLB_mse = mse(y_test, y_hat_rlb)
    Cte_mse = mse(y_test, y_hat_cte)

    #print(regresion_lineal.model)

    # Grafico el dataset y las predicciones
    # tomamos un vector equiespaciado de diez valores para graficar los modelos
    x_plot = np.linspace(0, 10, 10)
    y_plot_rl = regresion_lineal.model * x_plot
    y_plot_rlb = regresion_lineal_with_b.model[0] * x_plot + regresion_lineal_with_b.model[1]
    y_plot_cte = y_hat_cte

    plt.scatter(x_train, y_train, color='b', label='dataset')
    plt.plot(x_plot,y_plot_rl, color='m', label=f'Reg. Lineal  MSE:{RL_mse:.3f}')
    plt.plot(x_plot,y_plot_rlb, color='r', label=f'Reg. Lineal con b MSE:{RLB_mse:.3f}')
    plt.plot(x_test,y_plot_cte, color='g', label=f'F. Constante MSE:{Cte_mse:.3f}')
    plt.legend()
    plt.show()

