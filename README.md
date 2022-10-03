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
python xor.py
```

```
python mnist.py
```

---

## Supported Neural Layers
```python
Dense(numInputNeurons, numOutputNeurons) # For Weight Manipulation
Softmax() # For output percentage predictions

# Activation Functions
Sigmoid()
Tanh()
Relu()
LeakyRelu() # Leaky Relu Not Validated
```
## Network Class Usage

```python

# Network layers are initalized as list
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
saveNetwork(network, "mnistNetwork.pkl")

# Load the network from a file
network = loadNetwork("mnistNetwork.pkl")

```

## CUDA Support

This simple neural network library does provide support for Cupy to gain CUDA support (You must have CUDA Toolkit installed). This can be enabled by setting enableCuda in config.py to True. HOWEVER, enabling CUDA on smaller networks can actually DECREASE performance. Because copying data over to the GPU for calculation takes time. So unless you are training very large networks I would leave CUDA disabled.

## TODO

- Package And Host The Library
- Implement More Loss Functions
- Validate Leaky Relu Activation Function
- Implement Convolutional Layer
- Implement Flatten Layer
- Implement Dropout Layer
