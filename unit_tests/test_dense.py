import unittest

import sys
sys.path.append('..')
# Network Library Imports
from nn import *
import numpy as np
import math

class TestDenseLayer(unittest.TestCase):

    def test_dense_forward(self):
        dense_layer = Dense(1, 1)
        dense_layer.weights = np.arange(-2, 3, 1, dtype='float32')
        dense_layer.bias = np.full((1), 1, dtype='float32')
        input = np.arange(0.1, 0.6, 0.1, dtype='float32')
        # (0.1 * -2) + (0.2 * -1) + (0.3 * 0) + (0.4 * 1) + (0.5 * 2) + 1
        self.assertTrue(math.isclose(2.0, dense_layer.forward(input)))

    def test_dense_backwards(self):
        pass

    def test_dense_initialization(self):
        dense_layer = Dense(100, 10)
        self.assertTrue(dense_layer.weights.shape == (10, 100))
        self.assertTrue(dense_layer.bias.shape == (10, 1))
        self.assertTrue(dense_layer.layer_properties is not None)

if __name__ == '__main__':
    unittest.main()

#print(np.add(*np.indices((5, 5))))