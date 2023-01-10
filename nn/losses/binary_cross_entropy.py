import numpy as np
# Solves a niche error when the input to the log is zero.
from scipy.special import log1p

# https://neuralthreads.medium.com/binary-cross-entropy-loss-special-case-of-categorical-cross-entropy-loss-95c0c338d183
def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * log1p(y_pred) - (1 - y_true) * log1p(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)