import numpy as np


# Normalizar con z-score un dataset de nxm


def z_score_normal(dataset_X):
    val_med_col = np.mean(dataset_X, axis=0)
    s_desviation_col = np.std(dataset_X, axis=0)
    z_result = (dataset_X - val_med_col) / s_desviation_col

    return z_result


dataset = 10 * np.random.rand(10, 2)
# print(dataset)

print(" Z Score Normalization > {} ".format(z_score_normal(dataset)))
