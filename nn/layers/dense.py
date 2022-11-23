import numpy as np
from nn.layers.layer import Layer
from nn.layers.layer_properties import LayerProperties
from nn.initializers import *
from nn.optimizers import *

class Dense(Layer):
    # Default layer properties
    layer_properties = LayerProperties(0.05, Uniform(), Uniform(), SGD())

    def __init__(self, input_shape, output_shape, layer_properties: LayerProperties = None):
        self.weights = self.layer_properties.weight_initializer.get(output_shape, input_shape)
        self.bias = self.layer_properties.weight_initializer.get(output_shape, 1)

        # Optionally set the layer properties for all layers that utilize layer properties parameters
        if layer_properties is not None:
            # Replace all layer defaults with any non "None" layer properties.
            # This is just a lot of fancy code to allow you to override only 'some' of the default layer properties.
            # Instead of forcing you to populate all the parameters every time.
            for attr, value in layer_properties.__dict__.items():
                if getattr(layer_properties, attr) is not None:
                    setattr(self.layer_properties, attr, getattr(layer_properties, attr))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights += self.layer_properties.optimizer.calc(self.layer_properties.learning_rate, weights_gradient)
        self.bias += self.layer_properties.optimizer.calc(self.layer_properties.learning_rate, output_gradient)
        return input_gradient
    
    #Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.weights.shape[1]) + ", " + str(self.weights.shape[0]) + ")"