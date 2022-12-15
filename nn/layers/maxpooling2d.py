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
        self.depth = input_shape[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # padding supports various padding formats depending on how you want to pad the various sides.
        # 2
        # (2, 3)
        # ((1,2),(2,1))

    def forward(self, input):
        # I need to add the depth parameter and loop over it.
        #np.array([pool2d(channel, kernel_size=2, stride=2, padding=0, pool_mode='max') for channel in A])

        # Padding
        input = np.pad(input, self.padding, mode='constant')

        final_output_shape = (input.shape[0], 
                            (input.shape[1] - self.kernel_size) // self.stride[0] + 1,
                            (input.shape[2] - self.kernel_size) // self.stride[1] + 1)

        final_output = np.zeros(final_output_shape, dtype="float64")

        # I NEED TO CONVERT THIS INDEX MAP TO SOME KIND OF NUMPY ARRAY
        #self.index_map = [[[-1]*final_output_shape[2]]*final_output_shape[1]]*input.shape[0]
        self.index_map_x = np.zeros((input.shape[0], final_output_shape[1], final_output_shape[2]), dtype="int")
        self.index_map_y = np.zeros((input.shape[0], final_output_shape[1], final_output_shape[2]), dtype="int")

        for depth in range(input.shape[0]):

            # Window view of input array
            # Have to use input indexes 1 and 2 because index 0 is the depth.
            output_shape = ((input.shape[1] - self.kernel_size) // self.stride[0] + 1,
                            (input.shape[2] - self.kernel_size) // self.stride[1] + 1)
            
            # The array is 4D because it correlates each stride calculated with each output.
            # This is a pre-processing step so we can call the max function on each stride output and map them to the output shape.
            view_shape = (output_shape[0], output_shape[1], self.kernel_size, self.kernel_size)
            # The strides that will be applied to the input array
            view_strides = (self.stride[0] * input.strides[1], self.stride[1] * input.strides[2], input.strides[1], input.strides[2])
            
            # 4D array containing all the striding 2D outputs mapped to the 2D pooling output space.
            strided_result = as_strided(input, view_shape, view_strides)
            # print(strided_result.shape)
            # print(strided_result.shape)
                    
            # Result of max pooling along the kernel axes
            out = strided_result.max(axis=(2, 3))

            # print(out)

            # This index map contains the input indexes that the output gradient maps to.
            # Basically the value at index_map[x][y] is coordinates to the input that output_gradient[x][y] needs to be applied to.
            #self.index_map = np.zeros(output_shape, dtype="float64")
            
            # print(strided_result.shape[0])
            # print(strided_result.shape[1])
            for x in range(strided_result.shape[0]):
                for y in range(strided_result.shape[1]):
                    # I have the stride
                    # I have the stride output index (that's x and y)
                    # I just need a way to calculate the max value index (index local to the stride)
                    local_stride_index = np.unravel_index(np.argmax(strided_result[x][y], axis=None), strided_result[x][y].shape)
                    local_stride_index_x, local_stride_index_y = local_stride_index

                    #input_index = local_stride_index + (np.array(x,y) * self.stride)
                    #input_index = (local_stride_index_x + (x * self.stride[0]), local_stride_index_y + (y * self.stride[1]))
                    # replace x,y index with the above input_in
                    # print("{}, {}, {}".format(depth, x, y))
                    self.index_map_x[depth][x][y] = local_stride_index_x + (x * self.stride[0])
                    self.index_map_y[depth][x][y] = local_stride_index_y + (y * self.stride[1])
            final_output[depth] = out

        return final_output

    
    def backward(self, output_gradient):
        input_gradient = np.zeros(self.input_shape, dtype="float64")
        for depth in range(output_gradient.shape[0]):
            for x in range(0, output_gradient.shape[1]):
                for y in range(0, output_gradient.shape[2]):
                    input_x_index = self.index_map_x[depth][x][y]
                    input_y_index = self.index_map_y[depth][x][y]
                    input_gradient[depth][input_x_index][input_y_index] += output_gradient[depth][x][y]
        # print(input_gradient)
        # Currently prints a lot of input gradients that just have a bunch of zeros
        return input_gradient

    # Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "()"