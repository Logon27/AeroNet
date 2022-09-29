class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return forward propagation output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update weights and return input gradient
        pass