import numpy as np
from nn.layers.layer import Layer

class Dropout(Layer):
    def __init__(self, probability = 0.25):
        # Drop probability, if you want a success rate of 75% then you would
        # set a probability of 0.25 or 25% drop rate.
        self.probability = probability

    def forward(self, input):
        self.mask = np.random.binomial(1, self.probability, size=input.shape) / self.probability
        output = input * self.mask
        return output.reshape(input.shape)

    def backward(self, output_gradient):
        return output_gradient * self.mask

    # Modify string representation for network architecture printing
    def __str__(self):
        return self.__class__.__name__ + "({})".format(self.probability)