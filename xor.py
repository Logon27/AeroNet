import numpy as np
import matplotlib.pyplot as plt

from nn import *

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

#network = loadNetwork("xor_network.pkl")
network = Network(layers, mse, mse_prime, X, Y, X, Y, epochs=1000, learning_rate=0.05, batch_size=1)
# Train The Network
network.train()
saveNetwork(network, "xor_network.pkl")

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

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

ax.scatter(x, y, z, c=z, cmap="winter")
plt.show()