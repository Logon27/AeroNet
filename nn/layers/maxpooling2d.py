import numpy as np
from nn.layers.layer import Layer
from nn.initializers import *
from nn.optimizers import *
from typing import Tuple
from numpy.lib.stride_tricks import as_strided

# Modified from...
# https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy
# Helpful articles
# https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20#1f28
# https://deepnotes.io/maxpool

class MaxPooling2D(Layer):

    def __init__(self, input_shape, kernel_size: int, stride: Tuple[int,int]= (1,1), padding: Tuple[int,int]= (0,0)):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = input_shape[0]
        self.stride = stride
        # padding supports various padding formats depending on how you want to pad the various sides.
        # 2
        # (2, 3)
        # ((1,2),(2,1))
        self.padding = padding
        # The max pooling layer does not change the depth of the input.
        self.output_shape = (input_shape[0], 
                            (input_shape[1] - self.kernel_size) // self.stride[0] + 1,
                            (input_shape[2] - self.kernel_size) // self.stride[1] + 1)
        # This index map contains the input indexes that the output gradient maps to.
        # Basically the value at index_map[depth][x][y] is coordinates to the input that output_gradient[depth][x][y] needs to be applied to.
        self.index_map_x = np.zeros((self.depth, self.output_shape[1], self.output_shape[2]), dtype="int")
        self.index_map_y = np.zeros((self.depth, self.output_shape[1], self.output_shape[2]), dtype="int")

    def forward(self, input):
        # Padding
        input = np.pad(input, self.padding, mode='constant')
        
        # The array is 5D because it correlates each stride calculated with each output. (depth, output_x, output_y, stride_result_x, stride_result_y)
        # This is a pre-processing step so we can call the max function on each stride output and map them to the output shape.
        view_shape = (self.depth, self.output_shape[1], self.output_shape[2], self.kernel_size, self.kernel_size)
        # The strides that will be applied to the input array
        view_strides = (input.strides[0], self.stride[0] * input.strides[1], self.stride[1] * input.strides[2], input.strides[1], input.strides[2])
        
        # 5D array containing the depth and all the striding 2D outputs mapped to the 2D pooling output space.
        strided_result = as_strided(input, view_shape, view_strides)
                
        # Result of max pooling along the stride axes
        out = strided_result.max(axis=(3, 4))
        
        for depth in range(strided_result.shape[0]):
            for x in range(strided_result.shape[1]):
                for y in range(strided_result.shape[2]):
                    # Calculate the local index of the max value in each of the strides.
                    local_stride_index = np.unravel_index(np.argmax(strided_result[depth][x][y], axis=None), strided_result[depth][x][y].shape)
                    local_stride_index_x, local_stride_index_y = local_stride_index

                    # Calculate the input index based on the stride's local index, the strided_result array index, and the stride itself.
                    # Then store that input index to be used as a sort of reordering / mask for the output gradient in backprop.
                    self.index_map_x[depth][x][y] = local_stride_index_x + (x * self.stride[0])
                    self.index_map_y[depth][x][y] = local_stride_index_y + (y * self.stride[1])
        return out

    
    def backward(self, output_gradient):
        input_gradient = np.zeros(self.input_shape, dtype="float64")
        for depth in range(output_gradient.shape[0]):
            for x in range(output_gradient.shape[1]):
                for y in range(output_gradient.shape[2]):
                    input_x_index = self.index_map_x[depth][x][y]
                    input_y_index = self.index_map_y[depth][x][y]
                    # Map the input index tracked through the translation mask to the corresponding output gradient
                    # Only the max values used in the forward pass will receive the gradient in the backward pass.
                    # Since only those max values contributed to the error.
                    # I do a += here because there could be multiple gradient values mapped to a single input
                    # if the stride is smaller than the pooling size.
                    input_gradient[depth][input_x_index][input_y_index] += output_gradient[depth][x][y]
        return input_gradient

    # Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "()"