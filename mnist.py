from config import *
if enableCuda:
    import cupy as np
else:
    import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

# Network Library Imports
from network import Network
from dense import Dense
from activations import *
from losses import *
from fileio import *

def preprocess_data(x, y, limit):
    # Reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("longdouble") / 255
    # Encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

# Load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load clean MNIST copy for image display.
(x_train_image, y_train_image), (x_test_image, y_test_image) = mnist.load_data()

x_train, y_train = preprocess_data(x_train, y_train, 60000)
x_test, y_test = preprocess_data(x_test, y_test, 10000)

# Convert to cupy arrays if CUDA is enabled
if enableCuda:
    x_train, y_train = (np.asarray(x_train), np.asarray(y_train))
    x_test, y_test = (np.asarray(x_test), np.asarray(y_test))

# Network layers
layers = [
    Dense(28 * 28, 70),
    Sigmoid(),
    Dense(70, 35),
    Sigmoid(),
    Dense(35, 10),
    Softmax()
]

# network = loadNetwork("mnistNetwork.pkl")
network = Network(layers, mse, mse_prime, x_train, y_train, x_test, y_test, epochs=10, learning_rate=0.1, batch_size=1)
network.train()
# saveNetwork(network, "mnistNetwork.pkl")

#Visual Debug
fig, axes = plt.subplots(ncols=20, sharex=False, sharey=True, figsize=(20, 4))
for i in range(20):
    output = network.predict(x_test[i])
    prediction = np.argmax(output)
    #Convert to a string to prevent an error with cupy
    prediction = str(prediction)
    
    axes[i].set_title(prediction)
    axes[i].imshow(x_test_image[i], cmap='gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()