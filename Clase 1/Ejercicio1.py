import numpy as np


def norma_l0(mat):
    mask = mat >0
    norma = np.sum(mask)
    return norma


def norma_inf(mat):
    norma = np.max(mat,axis=1)
    return norma


def norma_l1(mat):
    abs_mat = np.abs(mat)
    norma = np.sum(abs_mat,axis=1)
    return norma


def norma_l2(mat):
    mat2 = np.power(mat,2)
    sum_mat2 = np.sum(mat2,axis=1)
    norma = np.power(sum_mat2,1/2)
    return norma


