import numpy as np
from nn.layers.layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape)

    # Modify string representation for network architecture printing
    def __str__(self):
        return self.__class__.__name__ + "({}, {})".format(self.input_shape, self.output_shape)