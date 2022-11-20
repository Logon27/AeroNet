import numpy as np
from nn.layers.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.uniform(-1, 1, size=(output_size, input_size))
        self.bias = np.random.uniform(-1, 1, size=(output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
    
    #Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.weights.shape[1]) + ", " + str(self.weights.shape[0]) + ")"