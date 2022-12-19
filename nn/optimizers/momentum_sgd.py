from .optimizer import Optimizer
import numpy as np

class MomentumSGD(Optimizer):
    def __init__(self, mu=0.90):
        self.mu = mu
        self.v = None
    
    def calc(self, learning_rate, gradient):
        if self.v is None:
            self.v = np.zeros(gradient.shape)
        self.v = self.mu * self.v + learning_rate * gradient
        return -self.v