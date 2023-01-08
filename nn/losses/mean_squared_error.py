import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mean_squared_error_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)