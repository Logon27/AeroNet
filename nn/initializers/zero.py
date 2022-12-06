import numpy as np
from .initializer import Initializer

class Zero(Initializer):
    def get(self, *shape):
        return np.full(shape, 0, dtype="float64")