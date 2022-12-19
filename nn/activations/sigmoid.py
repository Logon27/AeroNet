import numpy as np
from nn.activations.activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
    
    # Modify string representation for network architecture printing
    def __str__(self):
        return self.__class__.__name__ + "()"