from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self):
        pass

    def calc(self, learning_rate, gradient):
        return -learning_rate * gradient