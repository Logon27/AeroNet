from config import *
if enableCuda:
    import cupy as np
else:
    import numpy as np
from layer import Layer
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

# Both Relu activation functions are unverified at the moment.
class Relu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x >= 0, 1, 0)

        super().__init__(relu, relu_prime)

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

class Softmax(Layer):
    def forward(self, input):
        # More stable softmax
        #https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
        tmp = np.exp(input - max(input))
        self.output = tmp / np.sum(tmp)
        return self.output
    
    # Might need to re-evaluate this backprop after making the forward stable softmax
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
    
    def __str__(self):
        return self.__class__.__name__ + "()"