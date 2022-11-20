import numpy as np
from nn.activations.activation import Activation

class LeakyRelu(Activation):
    def __init__(self):
        def leaky_relu(x):
            alpha = 0.01
            return np.maximum(alpha * x, x)

        def leaky_relu_prime(x):
            alpha = 0.01
            dx = np.ones_like(x)
            dx[x < 0] = alpha
            return dx

        super().__init__(leaky_relu, leaky_relu_prime)
        
    def __str__(self):
        return self.__class__.__name__ + "()"