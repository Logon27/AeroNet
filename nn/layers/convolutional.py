import numpy as np
from scipy import signal
from nn.layers.layer import Layer
from nn.layers.layer_properties import LayerProperties
from nn.initializers import *
from nn.optimizers import *
from copy import deepcopy

class Convolutional(Layer):

    def __init__(self, input_shape, kernel_size, depth, bias_mode: str = 'untied', layer_properties: LayerProperties = None):
        # Default layer properties
        self.layer_properties = LayerProperties(learning_rate=0.05, weight_initializer=Uniform(), bias_initializer=Uniform(), optimizer=SGD())

        # Optionally set the layer properties for all layers that utilize layer properties parameters
        if layer_properties is not None:
            # Replace all layer defaults with any non "None" layer properties.
            # This is just a lot of fancy code to allow you to override only 'some' of the default layer properties.
            # Instead of forcing you to populate all the parameters every time.
            for attr, _ in layer_properties.__dict__.items():
                if getattr(layer_properties, attr) is not None:
                    # copy is necessary to ensure that individual layer classes don't get shared instances of an optimizer
                    # optimizers such as momentum sgd require separate instances to track velocity
                    setattr(self.layer_properties, attr, deepcopy(getattr(layer_properties, attr)))

        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_size = kernel_size
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = self.layer_properties.weight_initializer.get(*self.kernels_shape)

        valid_bias_modes = {'tied', 'untied'}
        if bias_mode not in valid_bias_modes:
            raise ValueError("bias_modes must be one of {}, but got bias_mode='{}'".format(valid_bias_modes, bias_mode))
        self.bias_mode = bias_mode

        if bias_mode == 'untied':
            self.biases = self.layer_properties.bias_initializer.get(*self.output_shape)
        elif bias_mode == 'tied':
            self.biases = self.layer_properties.bias_initializer.get(self.depth, 1, 1)

    def forward(self, input):
        if self.bias_mode == 'untied':
            return self.forward_untied(input)
        elif self.bias_mode == 'tied':
            return self.forward_tied(input)     
    
    def backward(self, output_gradient):
        if self.bias_mode == 'untied':
            return self.backward_untied(output_gradient)
        elif self.bias_mode == 'tied':
            return self.backward_tied(output_gradient)

    # Separated Untied and Tied implementations into their own functions
    # Untied bias: where you use use one bias per kernel and output
    # Tied bias: where you share one bias per kernel
    def forward_untied(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward_untied(self, output_gradient):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels += self.layer_properties.weight_optimizer.calc(self.layer_properties.learning_rate, kernels_gradient)
        self.biases += self.layer_properties.bias_optimizer.calc(self.layer_properties.learning_rate, output_gradient)
        return input_gradient
    
    def forward_tied(self, input):
        self.input = input
        self.output = np.zeros((self.depth, input.shape[1] - self.kernel_size + 1, input.shape[2] - self.kernel_size + 1))
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
            self.output[i] += self.biases[i]
        return self.output

    def backward_tied(self, output_gradient):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        self.kernels += self.layer_properties.weight_optimizer.calc(self.layer_properties.learning_rate, kernels_gradient)
        # Sum all the gradients for the output gradient of each kernel then multiply by the learning rate.
        self.biases += self.layer_properties.bias_optimizer.calc(self.layer_properties.learning_rate, np.apply_over_axes(np.sum, output_gradient, [1,2]))
        return input_gradient

    # Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "(" + "(" + str(self.input_shape[0]) + ", " + str(self.input_shape[1]) + ", " \
            + str(self.input_shape[2]) + ")"  + ", " + str(self.kernels_shape[3]) + ", " + str(self.depth) + ") bias_mode = " + self.bias_mode