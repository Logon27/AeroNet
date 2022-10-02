# PLEASE NOTE
# CUDA acceleration actually hurts performance for small examples like xor.py
# This is because it has to do a lot of setup and copying over to the GPU

from config import *
if enableCuda:
    import cupy as np
else:
    import numpy as np

import matplotlib.pyplot as plt

from network import Network
from dense import Dense
from activations import *
from losses import *

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
X = np.reshape(X, (4, 2, 1))
Y = np.reshape(Y, (4, 1, 1))

layers = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# Train The Network
network = Network(layers, mse, mse_prime, X, Y, X, Y, epochs=1000, learning_rate=0.05, batch_size=1)
network.train()

# Decision Boundary 3D Plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        inputArray = np.array([[x], [y]])
        z = network.predict(inputArray)
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

if enableCuda:
    # Cuda Enabled
    x = points[:, 0].get()
    y = points[:, 1].get()
    z = points[:, 2].get()
else:
    # Numpy Enabled
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

ax.scatter(x, y, z, c=z, cmap="winter")
plt.show()