import numpy as np
from .initializer import Initializer

class Zero(Initializer):
    def __init__(self):
        pass

    def get(self, *shape):
        return np.full(shape, 0, dtype="float64")