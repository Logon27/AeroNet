import numpy as np
from .initializer import Initializer

class Uniform(Initializer):
    def get(self, *shape, low=-1, high=1):
        return np.random.uniform(low, high, size=shape)