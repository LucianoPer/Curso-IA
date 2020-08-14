import numpy as np


def generar_aleatorios_conNan(dim, n_element, amplitud , cant_nan):
    dataset = amplitud * np.random.rand(n_element, dim)
    nan_index = np.random.choice(n_element*dim, cant_nan, replace=False)
    reshape_index = np.unravel_index(nan_index,dataset.shape)
    print(reshape_index)
    dataset[reshape_index] = np.nan
    return dataset


dataset = generar_aleatorios_conNan(2,10,10,3)


def reemplzar_nan_media(dataset):
    mean = np.nanmean(dataset,axis=0)
    dataset = np.where(np.isnan(dataset), mean, dataset)
    return dataset


print(dataset)
print(reemplzar_nan_media(dataset))


