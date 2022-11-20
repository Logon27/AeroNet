# Neural-Network-Library

## Creating A Virtual Environment

```
python -m venv venvNeuralNetworkLibrary
venvNeuralNetworkLibrary/scripts/activate.bat
pip install -r requirements.txt
```

## Testing Out A Network

Test your network against one of these datasets...
```
# move into the training examples directory
cd training_examples
# then execute one of the scripts below...
```

```
python xor.py
```

```
python mnist.py
```

```
# Convolutional neural network implementation for mnist
python mnist_conv.py
```

```
# Fully convolutional network implementation for mnist
python mnist_fcn.py
```
---

## Supported Neural Layers
```python
Dense(numInputNeurons, numOutputNeurons) # For weight manipulation
Convolutional((inputDepth, inputWidth, inputHeight), kernelSize, numKernels)
Reshape() # Modifies the shape of the numpy arrays passed between layers
Flatten() # Flattens a numpy array into a 2D matrix with a single column
Dropout(probability) # Randomly drops layer outputs based on a probability to prevent overfitting

# Activation Functions
Sigmoid()
Tanh()
Relu()
LeakyRelu() # Leaky Relu not validated
Softmax() # For output percentage predictions
```
## Network Class Usage

```python
# Import all neural network classes.
from nn import *

# Network layers are initalized as a list
network_layers = [
    Dense(28 * 28, 70),
    Sigmoid(),
    Dense(70, 35),
    Sigmoid(),
    Dense(35, 10),
    Softmax()
]

# Create a network object
network = Network(network_layers, loss_function, loss_function_prime, x_train_set, y_train_set, x_test_set, y_test_set, epochs=10, learning_rate=0.1, batch_size=1)

# Train the network
network.train()

# Generate a prediction from the network given a single input array
prediction_array = network.predict(input_array)

# Save the network to a file
saveNetwork(network, "mnist_network.pkl")

# Load the network from a file
network = loadNetwork("mnist_network.pkl")

```

## CUDA Support

This library used to support CUDA. However, it has since been removed because it was too hard to maintain with more complex layer implementations.

## TODO

- Package And Host The Library
- Implement More Loss Functions
- Validate Leaky Relu Activation Function
- Implement Max Pooling
- Implement Avg Pooling
- Implement Adaptive Avg Pooling
- Implement Batch Normalization
- Implement Swappable Optimizers
- Separate the Model into its own class
- Implement Layer Input Size Calculations