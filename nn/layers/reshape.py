import numpy as np
from nn.interfaces.layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

    # Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.input_shape) + ", " + str(self.output_shape) + ")"