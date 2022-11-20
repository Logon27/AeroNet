import numpy as np
from nn.activations.activation import Activation

class Relu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x >= 0, 1, 0)

        super().__init__(relu, relu_prime)
     
    def __str__(self):
        return self.__class__.__name__ + "()"