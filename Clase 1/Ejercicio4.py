import numpy as np


class Precision():
    def __call__(self, prediction,truth):
        # Para TP hacemos un and logico entre prediction =1 y truth =1
        TP_mask = (prediction == 1) & (truth == 1)
        # Sumamos los 1 de TP
        TP = TP_mask.sum()
        # Hago lo mismo para el calculo de FP
        FP_mask = (prediction == 1) & (truth == 0)
        FP = FP_mask.sum()
        return TP/(TP+FP)


class Recall():
    def __call__(self, prediction,truth):
        TP_mask = (prediction == 1) & (truth == 1)
        TP = TP_mask.sum()
        FN_mask = (prediction == 0) & (truth == 1)
        FN = FN_mask.sum()
        return TP / (TP + FN)


class Accuracy():
    def __call__(self, prediction,truth):
        TP_mask = (prediction == 1) & (truth == 1)
        TP = TP_mask.sum()
        TN_mask =  (prediction == 0) & (truth == 0)
        TN = TN_mask.sum()
        FP_mask = (prediction == 1) & (truth == 0)
        FP = FP_mask.sum()
        FN_mask = (prediction == 0) & (truth == 1)
        FN = FN_mask.sum()
        return (TP+TN)/(TP+TN+FP+FN)

    