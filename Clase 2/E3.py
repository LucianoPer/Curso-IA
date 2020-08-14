# Implementacion  de Kmeans con Numpy

import numpy as np


def k_means(dataset, n_clusters, cant_iterat=10):
    # creo una matriz indice random de de 1 x n_clusters
    n_indexes = np.random.randint(0, dataset.shape[0], n_clusters)
    print(n_indexes)
    # tomo los centroides iniciales en los indices aleatorios
    centroids = dataset[n_indexes]
    for i in range(cant_iterat):
        centroids, cluster_ids = k_means_loop(dataset, centroids)
    return centroids, cluster_ids


def k_means_loop(X, centroids):
    expanded_centroids = centroids[:, None]
    distances = np.sqrt(np.sum((expanded_centroids - X) ** 2, axis=2))
    arg_min = np.argmin(distances, axis=0)

    for i in range(centroids.shape[0]):
        centroids[i] = np.mean(X[arg_min == i, :], axis=0)

    return centroids, arg_min