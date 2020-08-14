import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans



def generar_dataset(dim , n_elements , n_clusters, dist_cluster):
    clusters_cent = np.random.rand(n_clusters,dim)
    # Repetimos los rand generados como clusters tantas veces hasta el numero total de elementos y los distanciamos
    clusters = dist_cluster * np.repeat(clusters_cent, int(n_elements/n_clusters), axis=0)
    # agregamos ruido a los valores generdos
    noise = np.random.normal(loc=0,scale=1,size=(clusters.size//dim,dim))
    clusters_res = clusters + noise
    return  clusters_res


def cambiar_valores_Nan(dataset , cant_nan):
    nan_index = np.random.choice(dataset.size, cant_nan, replace=False)
    reshape_index = np.unravel_index(nan_index,dataset.shape)
    print(reshape_index)
    dataset[reshape_index] = np.nan
    return dataset


def reemplzar_nan_media(dataset):
    mean = np.nanmean(dataset,axis=0)
    dataset = np.where(np.isnan(dataset), mean, dataset)
    return dataset


def get_media_feature(dataset):
    return np.mean(dataset, axis=0)


def get_std_feature(dataset):
    return np.std(dataset,axis=0)


def get_norma_l2_feature(dataset):
    return np.sqrt(np.sum(dataset**2,axis=0))

def exponential(x, lambda_param=1.0):
    return (- np.log(1-x) / lambda_param)


def my_PCA_function(dataset_X, nueva_dim):
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
    return np.matmul(X,v[:,:nueva_dim])



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




if __name__ == '__main__':
    n_elementos = 1000
    dim = 4
    n_clusters = 4
    cant_nan = n_elementos * 0.01

    # generamos el dataset
    data = generar_dataset(dim,n_elementos,n_clusters,50)

    # Cambiamos el 1% de valores a nan
    cambiar_valores_Nan(data,int(cant_nan))

    # Guardamos el resultado en un pkl
    np.save('data_clase2.npy', data , allow_pickle=True)

    # Carga del dataset pickle
    data_pikle = np.load('data_clase2.npy',allow_pickle=True)

    # Reemplazo los nan por la media
    data = reemplzar_nan_media(data)

    # media y desviacion y norma l2
    print("Media de los features ",get_media_feature(data))
    print("Desviacion :",get_std_feature(data))
    print("Norma l2 :",get_norma_l2_feature(data))

    # expando dataset con columna con variable alatoria exponencial a todos los puntos
    #exponential()
    #plt.hist(exp_vector)
    #plt.show()

    # Reduccion por PCA
    data_pca = my_PCA_function(data,2)

    # Calculo de Kmeans para los clusters
    #kmeans_skl = KMeans(n_clusters,random_state=0).fit(data_pca)
    centroid, clus_ids2 = k_means(data_pca,n_clusters)
    plt.scatter(data_pca[:,0],data_pca[:,1])
    plt.scatter(centroid[:,0],centroid[:,1])
    #plt.scatter(kmeans_skl.cluster_centers_[:,0],kmeans_skl.cluster_centers_[:,1])
    plt.show()