import numpy as np
from .initializer import Initializer

class Uniform(Initializer):
    def __init__(self):
        pass

    def get(self, *shape):
        return np.random.uniform(-1, 1, size=shape)