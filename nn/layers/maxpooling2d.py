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

    def __init__(self, kernel_size: int, stride: Tuple[int,int]= (1,1), padding: Tuple[int,int]= (0,0)):
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
        self.index_map = [[0]*output_shape[0]]*output_shape[1]
        # print(strided_result.shape[0])
        # print(strided_result.shape[1])
        for x in range(strided_result.shape[0]):
            for y in range(strided_result.shape[1]):
                # I have the stride
                # I have the stride output index (that's x and y)
                # I just need a way to calculate the max value index (index local to the stride)
                local_stride_index = np.unravel_index(np.argmax(strided_result[x][y], axis=None), strided_result[x][y].shape)
                local_stride_index_x, local_stride_index_y = local_stride_index

                local_stride_index_x + (x * self.stride[0]), local_stride_index_y + (y * self.stride[1])
                #input_index = local_stride_index + (np.array(x,y) * self.stride)
                input_index = (local_stride_index_x + (x * self.stride[0]), local_stride_index_y + (y * self.stride[1]))

                # replace x,y index with the above input_in
                self.index_map[x][y] = input_index



        # Each of these outputs maps to a single index in the input array that contributed to the error.
        
        # Basically I need to create a 2d input gradient array and map all the output_gradient errors to the input gradient array.
        # If the same input contributed to the error more than once due to pooling overlap then just sum the values.
        # That will give me the input gradient, but I still need to figure out the kernel and bias gradient.
        return out

    
    def backward(self, output_gradient):
        print('----------')
        print(output_gradient.shape)
        input_gradient = np.zeros(output_gradient.shape, dtype="float64")
        for x in range(0, output_gradient[0].shape):
            for y in range(0, output_gradient[1].shape):
                input_x_index, input_y_index = self.index_map[x][y]
                input_gradient[input_x_index][input_y_index] = output_gradient[x][y]
        return input_gradient

    # Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "()"