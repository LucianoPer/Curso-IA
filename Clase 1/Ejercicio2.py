import numpy as np
from Ejercicio1 import norma_l2


def sorting_l2(mat):
    mat_l2 = norma_l2(mat)
    mat_sort_index = np.argsort(mat_l2)
    mat_sort_l2 = mat[mat_sort_index,:]
    return mat_sort_l2[::-1]
