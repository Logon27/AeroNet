from .optimizer import Optimizer

class SGD(Optimizer):
    def calc(self, learning_rate, gradient):
        return -learning_rate * gradient