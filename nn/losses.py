import numpy as np
# Solves a niche error when the input to the log is zero.
from scipy.special import log1p

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

# https://neuralthreads.medium.com/binary-cross-entropy-loss-special-case-of-categorical-cross-entropy-loss-95c0c338d183
def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * log1p(y_pred) - (1 - y_true) * log1p(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

# https://neuralthreads.medium.com/categorical-cross-entropy-loss-the-most-important-loss-function-d3792151d05b
def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 10**-100))

def categorical_cross_entropy_prime(y_true, y_pred):
    return -y_true/(y_pred + 10**-100)