# Aplicacion de la inversa generalizada de una funcion especifica
import numpy as np

def exponentian_random_variable(lambda_param,size):
    # generamos una variable aleatoria uniforme
    uniform_rand_var = np.random.uniform(low=0.0,high=1.0,size=size)
    return (-1 / lambda_param) * np.log(1-uniform_rand_var)


