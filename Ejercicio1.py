import numpy as np


def norma_l0(mat):
    mask = mat != 0
    norma = len(mat[mask])
    return norma


def norma_inf(mat):
    vec = np.sort(mat, axis=None)
    norma = vec[-1]
    return norma


def norma_l1(mat):


m = np.array([[1, -2, 1, 0], [0, 5, 1, 4]])
print("Norma l0: {}".format(norma_l0(m)))
print("Norma infinita: {}".format(norma_inf(m)))