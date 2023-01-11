import numpy as np
from nn.layers.layer import Layer
from nn.initializers import *
from nn.optimizers import *
from typing import Tuple
from numpy.lib.stride_tricks import as_strided

# Helpful reference articles
# https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy
# https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20#1f28
# https://deepnotes.io/maxpool

# In the future I would ideally like to use the numpy array directly to map the output gradient indices.
# https://stackoverflow.com/questions/26194389/how-to-rearrange-array-based-upon-index-array
# I feel like it would probably make this class more efficient

# The max pooling layer does not change the depth of the input. And it has no trainable parameters.
class MaxPooling2D(Layer):

    def __init__(self, input_shape, kernel_size, stride: Tuple[int,int]= (2,2), padding = (0)):
        self.input_shape = input_shape
        self.kernel_size = self.parse_kernel(kernel_size)
        self.stride = stride
        self.padding = self.parse_padding(padding)
        
        # The final output shape of the MaxPooling layer
        self.output_shape = (self.input_shape[0], 
                            (self.input_shape[1] - self.kernel_size[0] + (self.padding[1][0] + self.padding[1][1])) // self.stride[0] + 1,
                            (self.input_shape[2] - self.kernel_size[1] + (self.padding[2][0] + self.padding[2][1])) // self.stride[1] + 1)
        # This index map contains the input indexes that the output gradient maps to.
        # Basically the value at index_map[depth][x][y] is coordinates to the input that output_gradient[depth][x][y] needs to be applied to.
        self.index_map_x = np.zeros((self.output_shape[0], self.output_shape[1], self.output_shape[2]), dtype="int")
        self.index_map_y = np.zeros((self.output_shape[0], self.output_shape[1], self.output_shape[2]), dtype="int")

        # Run sanity checks to eliminate input errors
        self.sanity_checks()

    def forward(self, input):
        # Padding. Have to pad with negative infinity for max pooling.
        # Otherwise padding could be identified as the "max" if the inputs are negative values.
        input = np.pad(input, self.padding, mode='constant', constant_values=np.NINF)
        
        # The array is 5D because it correlates each stride calculated with each output. (depth, output_x, output_y, stride_result_x, stride_result_y)
        # This is a pre-processing step so we can call the max function on each stride output and map them to the output shape.
        view_shape = (self.output_shape[0], self.output_shape[1], self.output_shape[2], self.kernel_size[0], self.kernel_size[1])
        # The strides that will be applied to the input array
        view_strides = (input.strides[0], self.stride[0] * input.strides[1], self.stride[1] * input.strides[2], input.strides[1], input.strides[2])
        
        # 5D array containing the depth and all the striding 2D outputs mapped to the 2D pooling output space.
        strided_result = as_strided(input, view_shape, view_strides)
                
        # Result of max pooling along the stride axes
        out = strided_result.max(axis=(3, 4))

        # Flatten the last two array dimensions because argmax only supports flat arrays. Now a 4D array.
        argmax_arr = np.reshape(strided_result, (strided_result.shape[0], strided_result.shape[1], strided_result.shape[2], strided_result.shape[3] * strided_result.shape[4]))
        # Find the index of the max value on the flat array (axis 3). argmax only takes the index of the first max value if there are duplicates.
        argmax_arr = np.argmax(argmax_arr, axis=3)

        # Calculate the input index based on the stride's local index, the strided_result array index, and the stride itself.
        # Then store that input index to be used as a sort of reordering / mask for the output gradient in backprop.
        # self.index_map_x[depth][x][y] = local_stride_index_x + (strided_result_x_index * self.stride[0])
        # self.index_map_y[depth][x][y] = local_stride_index_y + (strided_result_y_index * self.stride[1])

        # indices gives me a mask for the stride's coordinates in the output array. See the link below for examples...
        # https://numpy.org/doc/stable/reference/generated/numpy.indices.html
        indices = np.indices((strided_result.shape[1], strided_result.shape[2]))

        # Make a copy of the argmax_arr because we will need two copies.
        # One for the x coordinates and one for the y coordinates.
        argmax_arr_x = argmax_arr.copy()
        # Floor divide by the stride's row size (number of columns) to get the x coordinate
        argmax_arr_x = argmax_arr_x // self.stride[1]
        # Modulus by the stride's row size (number of columns) to get the y coordinate. (uses the original argmax_arr for efficiency instead of another copy)
        argmax_arr_y = argmax_arr % self.stride[1]
        # Multiply each indice mask by the cooresponding row or column size
        indices_x = indices[0] * self.stride[0]
        indices_y = indices[1] * self.stride[1]
        # Loop through the depth of the input and apply the mask calculation to each input depth
        for depth in range(self.output_shape[0]):
            # Final calculation of the local_stride_index added with the (stride output coordinate * stride size)
            self.index_map_x[depth] = argmax_arr_x[depth] + indices_x
            self.index_map_y[depth] = argmax_arr_y[depth] + indices_y
            # Elementwise subtract top and left padding from the indices.
            # This is because we calculated the input index for the padded input and not the original input.
            self.index_map_x[depth] = self.index_map_x[depth] - self.padding[1][0]
            self.index_map_y[depth] = self.index_map_y[depth] - self.padding[2][0]
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

    # padding supports two padding formats depending on how you want to pad the various sides.
    # 2 or (2). Pads all sides
    # ((top, bottom), (left, right))
    # ((1,2),(2,1)). Pads sides individually
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
        return self.__class__.__name__ + "(({}, {}, {}), kernel_size = {}, stride = {}, padding = {})".format(
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.kernel_size,
            self.stride,
            self.padding[1:]
        )