import sys
sys.path.append('..')

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

# Network Library Imports
from nn import *

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

# Network layers
layers = [
    Dense(28 * 28, 70),
    Sigmoid(),
    Dense(70, 35),
    Sigmoid(),
    Dense(35, 10),
    Softmax()
]

# network = loadNetwork("mnist_network.pkl")
network = Network(layers, TrainingSet(x_train, y_train, x_test, y_test, np.argmax), loss=mse, loss_prime=mse_prime, epochs=10, batch_size=1)
network.train()
saveNetwork(network, "mnist_network.pkl")

# Visual Debug After Training
rows = 5
columns = 10
fig, axes = plt.subplots(nrows=rows, ncols=columns, sharex=False, sharey=True, figsize=(12, 8))
fig.canvas.manager.set_window_title('Network Predictions')
# "i" represents the test set starting index.
i = 0
for j in range(rows):
    for k in range(columns):
        output = network.predict(x_test[i])
        prediction = np.argmax(output)
        # Convert to a string to prevent an error with cupy
        prediction = str(prediction)
        axes[j][k].set_title(prediction)
        axes[j][k].imshow(x_test_image[i], cmap='gray')
        axes[j][k].get_xaxis().set_visible(False)
        axes[j][k].get_yaxis().set_visible(False)
        i += 1
plt.show()