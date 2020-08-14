import numpy as np


def build_cluster(n_samples , separacion):
    centroids = np.array([
        [1,0,0,0],
        [0,1,0,0],
    ],dtype=np.float32)
    centroids = centroids * separacion
    # creamos la mitad de los valores iguales a los centroides
    data = np.repeat(centroids,n_samples/2,axis=0)
    # agregamos ruido blanco con media 0 y desviacion 1
    normal_noise = np.random.normal(loc=0,scale=1,size=(n_samples,4))
    # sumamos el ruido blanco a los valores creados (element wise)
    data = data + normal_noise
    cluster_ids = np.array([
        [0],
        [1],
    ])
    cluster_ids = np.repeat(cluster_ids,n_samples/2,axis=0)
    return data, cluster_ids

