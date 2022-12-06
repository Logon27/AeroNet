import numpy as np
from .initializer import Initializer

class Uniform(Initializer):
    def get(self, *shape):
        return np.random.uniform(-1, 1, size=shape)