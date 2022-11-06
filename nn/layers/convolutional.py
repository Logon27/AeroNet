import numpy as np
from scipy import signal
from nn.interfaces.layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth, bias_mode: str = 'untied'):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_size = kernel_size
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.uniform(-1, 1, size=self.kernels_shape)

        valid_bias_modes = {'tied', 'untied'}
        if bias_mode not in valid_bias_modes:
            raise ValueError("bias_modes must be one of {}, but got bias_mode='{}'".format(valid_bias_modes, bias_mode))
        self.bias_mode = bias_mode

        if bias_mode == 'untied':
            self.biases = np.random.uniform(-1, 1, size=self.output_shape)
        elif bias_mode == 'tied':
            self.biases = np.full((self.depth, 1, 1), 0, dtype="float64")

    def forward(self, input):
        if self.bias_mode == 'untied':
            return self.forward_untied(input)
        elif self.bias_mode == 'tied':
            return self.forward_tied(input)     
    
    def backward(self, output_gradient, learning_rate):
        if self.bias_mode == 'untied':
            return self.backward_untied(output_gradient, learning_rate)
        elif self.bias_mode == 'tied':
            return self.backward_tied(output_gradient, learning_rate)

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

    def backward_untied(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    
    def forward_tied(self, input):
        self.input = input
        self.output = np.zeros((self.depth, input.shape[1] - self.kernel_size + 1, input.shape[2] - self.kernel_size + 1))
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
            self.output[i] += self.biases[i]
        return self.output

    def backward_tied(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        self.kernels -= learning_rate * kernels_gradient
        # Sum all the gradients for the output gradient of each kernel then multiply by the learning rate.
        self.biases -= learning_rate * np.apply_over_axes(np.sum, output_gradient, [1,2])
        return input_gradient

    # Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "(" + "(" + str(self.input_shape[0]) + ", " + str(self.input_shape[1]) + ", " \
            + str(self.input_shape[2]) + ")"  + ", " + str(self.kernels_shape[3]) + ", " + str(self.depth) + ") bias_mode = " + self.bias_mode