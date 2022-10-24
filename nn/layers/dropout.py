import numpy as np
from nn.interfaces.layer import Layer

class Dropout(Layer):
    def __init__(self, probability = 0.25):
        self.probability = probability

    def forward(self, input):
        self.mask = np.random.binomial(1, self.probability, size=input.shape) / self.probability
        output = input * self.mask
        return output.reshape(input.shape)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask

    #Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.probability) +  ")"