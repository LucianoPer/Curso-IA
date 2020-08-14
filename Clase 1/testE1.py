from unittest import TestCase
import numpy as np
from Ejercicio1 import norma_l0


class NormaTest(TestCase):
    def test_indexer(self):
        mat = np.array([[1,2,-1],[-1,5,0]])
        expected = 4
        res = norma_l0(mat)
        print(res)
        self.assertTrue(expected,res)


