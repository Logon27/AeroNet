import numpy as np
from scipy import signal
from nn.layers.layer import Layer
from nn.layers.layer_properties import LayerProperties
from nn.initializers import *
from nn.optimizers import *
from copy import deepcopy
from typing import Tuple

class Convolutional(Layer):

    def __init__(self, input_shape, kernel_size, kernel_depth: int, stride: Tuple[int,int]= (1,1), padding = (0), bias_mode: str = 'untied', layer_properties: LayerProperties = None):
        # Default layer properties
        self.layer_properties = LayerProperties(learning_rate=0.05, weight_initializer=Uniform(), bias_initializer=Uniform(), optimizer=SGD())

        # Parse and update any optional layer properties
        self.parse_layer_properties(layer_properties)

        self.kernel_depth = kernel_depth
        self.input_shape = input_shape
        self.input_depth = input_shape[0]
        self.kernel_size = self.parse_kernel(kernel_size)
        self.kernels_shape = (self.kernel_depth, self.input_depth, self.kernel_size[0], self.kernel_size[1])
        self.kernels = self.layer_properties.weight_initializer.get(*self.kernels_shape)

        self.stride = stride
        # Parse the padding into a 3D tuple ((padding_height1, padding_width1), (padding_height2, padding_width2), (padding_height2, padding_width2))
        # One tuple pair for each dimension. (depth, kernel height, kernel width)
        self.padding = self.parse_padding(padding)

        self.output_shape = (self.kernel_depth,
                            (self.input_shape[1] - self.kernel_size[0] + (self.padding[1][0] + self.padding[1][1])) // self.stride[0] + 1,
                            (self.input_shape[2] - self.kernel_size[1] + (self.padding[2][0] + self.padding[2][1])) // self.stride[1] + 1)

        valid_bias_modes = {'tied', 'untied'}
        if bias_mode not in valid_bias_modes:
            raise ValueError("bias_mode must be one of {}, but got bias_mode='{}'".format(valid_bias_modes, bias_mode))
        self.bias_mode = bias_mode

        if bias_mode == 'untied':
            self.biases = self.layer_properties.bias_initializer.get(*self.output_shape)
        elif bias_mode == 'tied':
            self.biases = self.layer_properties.bias_initializer.get(self.kernel_depth, 1, 1)
        
        # Run sanity checks to eliminate input errors
        self.sanity_checks()

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
        input = np.pad(input, self.padding, mode='constant')
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.kernel_depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(input[j], self.kernels[i, j], "valid")[::self.stride[0], ::self.stride[1]]
        return self.output

    def backward_untied(self, output_gradient):
        # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
        # link to the dilation concept is above
        # The output gradient must be dilated before it can be correlated with the input due to striding.
        # The 1 * min is basically just a toggle. Because in the case of a stride of 1 you do NOT want to subtract 1 or dilate the output.
        depth, height, width = output_gradient.shape
        dilated_output_gradient = np.zeros((depth,
                        height + ((self.stride[0] - 1) * height - (1 * min(self.stride[0] - 1, 1))),
                        width + ((self.stride[1] - 1) * width - (1 * min(self.stride[1] - 1, 1)))),
                        dtype=output_gradient.dtype)
        dilated_output_gradient[::, ::self.stride[0], ::self.stride[1]] = output_gradient

        # If there is a non zero padding of we need to crop the output_gradient to match the input gradient.
        # This results in minor data loss but is easier to implement.
        # [top:bottom, left:right]
        dilated_trunc_output_gradient = dilated_output_gradient
        if self.padding[1][0] != 0:
            dilated_trunc_output_gradient = dilated_trunc_output_gradient[:, self.padding[1][0]:, :]
        if self.padding[1][1] != 0:
            dilated_trunc_output_gradient = dilated_trunc_output_gradient[:, :-self.padding[1][1], :]
        if self.padding[2][0] != 0:
            dilated_trunc_output_gradient = dilated_trunc_output_gradient[:, :, self.padding[2][0]:]
        if self.padding[2][1] != 0:
            dilated_trunc_output_gradient = dilated_trunc_output_gradient[:, :, :-self.padding[2][1]]

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.kernel_depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], dilated_output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(dilated_trunc_output_gradient[i], self.kernels[i, j], "full")

        self.kernels += self.layer_properties.weight_optimizer.calc(self.layer_properties.learning_rate, kernels_gradient)
        self.biases += self.layer_properties.bias_optimizer.calc(self.layer_properties.learning_rate, output_gradient)
        return input_gradient
    
    def forward_tied(self, input):
        input = np.pad(input, self.padding, mode='constant')
        self.input = input
        self.output = np.zeros(self.output_shape)
        for i in range(self.kernel_depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")[::self.stride[0], ::self.stride[1]]
            self.output[i] += self.biases[i]
        return self.output

    def backward_tied(self, output_gradient):
        depth, height, width = output_gradient.shape
        dilated_output_gradient = np.zeros((depth,
                        height + ((self.stride[0] - 1) * height - (1 * min(self.stride[0] - 1, 1))),
                        width + ((self.stride[1] - 1) * width - (1 * min(self.stride[1] - 1, 1)))),
                        dtype=output_gradient.dtype)
        dilated_output_gradient[::, ::self.stride[0], ::self.stride[1]] = output_gradient

        # If there is a non zero padding of we need to crop the output_gradient
        # [top:bottom, left:right]
        dilated_trunc_output_gradient = dilated_output_gradient
        if self.padding[1][0] != 0:
            dilated_trunc_output_gradient = dilated_trunc_output_gradient[:, self.padding[1][0]:, :]
        if self.padding[1][1] != 0:
            dilated_trunc_output_gradient = dilated_trunc_output_gradient[:, :-self.padding[1][1], :]
        if self.padding[2][0] != 0:
            dilated_trunc_output_gradient = dilated_trunc_output_gradient[:, :, self.padding[2][0]:]
        if self.padding[2][1] != 0:
            dilated_trunc_output_gradient = dilated_trunc_output_gradient[:, :, :-self.padding[2][1]]

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.kernel_depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], dilated_output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(dilated_trunc_output_gradient[i], self.kernels[i, j], "full")
        self.kernels += self.layer_properties.weight_optimizer.calc(self.layer_properties.learning_rate, kernels_gradient)
        # Sum all the gradients for the output gradient of each kernel then multiply by the learning rate.
        self.biases += self.layer_properties.bias_optimizer.calc(self.layer_properties.learning_rate, np.sum(output_gradient, keepdims=True, axis=(1,2)))
        return input_gradient
    
    def parse_kernel(self, kernel):
        # checks for single integer
        if isinstance(kernel, int):
            return (kernel, kernel)
        # checks for single value in tuple
        elif isinstance(kernel, tuple) and len(kernel) == 1:
            return (kernel[0], kernel[0])
        elif isinstance(kernel, tuple) and len(kernel) == 2:
            # No changes necessary if tuple with 2 values.
            return kernel
        else:
            raise ValueError("Kernel size invalid. Please use one of the following formats...\n" \
                            "kernel_size = 3\n" \
                            "kernel_size = (3)\n" \
                            "kernel_size = (3,3)\n")
    
    def parse_padding(self, padding):
        # checks for single integer
        if isinstance(padding, int):
            # The extra tuple of zeros is for the depth dimension which needs no padding.
            return ((0, 0), (padding, padding), (padding, padding))
        # checks for single value in tuple
        elif isinstance(padding, tuple) and len(padding) == 1:
            return ((0, 0), (padding[0], padding[0]), (padding[0], padding[0]))
        elif isinstance(padding, tuple) and len(padding) == 2:
            # (height, width) or (3, 3)
            if isinstance(padding[0], int) and isinstance(padding[1], int):
                return ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
            # ((top, bottom), (left, right))
            elif isinstance(padding[0], tuple) and isinstance(padding[1], tuple) and len(padding[0]) == 2 and len(padding[1]) == 2:
                return ((0, 0), (padding[0][0], padding[0][1]), (padding[1][0], padding[1][1]))
        else:
            raise ValueError("Padding size invalid. Please use one of the following formats...\n" \
                            "padding = 3\n" \
                            "padding = (3)\n" \
                            "padding = (3,3)\n" \
                            "padding = ((3,3), (3,3))\n")
    
    def parse_layer_properties(self, layer_properties):
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

    def sanity_checks(self):
        # Error checking to make sure the kernel and stride can evenly map across the input.
        dimension_1 = (self.input_shape[1] - self.kernel_size[0] + (self.padding[1][0] + self.padding[1][1])) / self.stride[0] + 1
        dimension_2 = (self.input_shape[2] - self.kernel_size[1] + (self.padding[2][0] + self.padding[2][1])) / self.stride[1] + 1
        if not dimension_1.is_integer() or not dimension_2.is_integer():
            raise ValueError("(input_size - kernel_size + (2 * padding)) / stride ... must be an integer.\n" \
                            "dimension 0 = depth\n" \
                            "dimension 1 = input height\n" \
                            "dimension 2 = input width\n" \
                            "This library does not handle cropping inputs for kernels / strides that do not evenly divide the input.")
        
        # If the padding is larger than the kernel size then you have values that will always be zero.
        # Because the output will only be calculated based on padding and not any input data
        if self.padding[1][0] >= self.kernel_size[0]:
            raise ValueError("Padding size {} is greater than or equal to kernel size {}".format(self.padding[1][0], self.kernel_size))
        elif self.padding[1][1] >= self.kernel_size[0]:
            raise ValueError("Padding size {} is greater than or equal to kernel size {}".format(self.padding[1][1], self.kernel_size))
        elif self.padding[2][0] >= self.kernel_size[1]:
            raise ValueError("Padding size {} is greater than or equal to kernel size {}".format(self.padding[2][0], self.kernel_size))
        elif self.padding[2][1] >= self.kernel_size[1]:
            raise ValueError("Padding size {} is greater than or equal to kernel size {}".format(self.padding[2][1], self.kernel_size))

    # Modify string representation for network architecture printing
    def __str__(self):
        return self.__class__.__name__ + "(({}, {}, {}), kernel_size = {}, kernel_depth = {}, bias_mode = {}, stride = {}, padding = {})".format(
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.kernel_size,
            self.kernel_depth,
            self.bias_mode,
            self.stride,
            self.padding[1:]
        )