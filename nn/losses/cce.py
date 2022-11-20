import numpy as np

# https://neuralthreads.medium.com/categorical-cross-entropy-loss-the-most-important-loss-function-d3792151d05b
def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 10**-100))

def categorical_cross_entropy_prime(y_true, y_pred):
    return -y_true/(y_pred + 10**-100)