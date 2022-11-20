import numpy as np
from nn.layers.layer import Layer

class Softmax(Layer):
    def forward(self, input):
        # More stable softmax
        #https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
        tmp = np.exp(input - max(input))
        self.output = tmp / np.sum(tmp)
        return self.output
    
    # Might need to re-evaluate this backprop after making the forward stable softmax
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
    
    def __str__(self):
        return self.__class__.__name__ + "()"