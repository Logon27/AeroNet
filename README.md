# Neural-Network-Library

## Creating A Virtual Environment

Python 3.10 is recommended for using this neural network library

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
network = Network(network_layers, TrainingSet(input_train, output_train, input_test, output_test, post_processing), \
    loss_function, loss_function_prime, epochs=10, batch_size=1)

# Train the network
network.train()

# Generate a prediction from the network given a single input array
prediction_array = network.predict(input_array)

# Save the network to a file
saveNetwork(network, "mnist_network.pkl")

# Load the network from a file
network = loadNetwork("mnist_network.pkl")
```

## TrainingSet Explained

The TrainingSet object is simply an object that stores your training and validation set data for your network. As well as a post processing function that is applied to your network predictions and training set data before comparisons are made.

```python
TrainingSet(input_train, output_train, input_test, output_test, post_processing)
```

input_train = training set inputs
output_train = training set outputs
input_test = test / valdation set inputs
output_test = test / valdation set outputs

post_processing is a function applied to your network prediction and training set data before they are compared. For example with the xor problem your network makes predictions based on floats. So you may want to round to the nearest float before comparing the prediction to the desired output. So you could pass np.rint to round to the nearest in. This allows you to get accurate test accuracy output during training. The same applies to things like mnist. You may have a softmax output with percentage predictions based on 10 indices (0 through 9). You may want your prediction to be the indice with the highest percentage. Therefore, you could pass the np.argmax function as the post processing function. Hopefully you can see why this is useful.

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

## Supported Optimizers

```python
SGD() # Stochastic Gradient Descent
```

## Supported Initializers

```python
Uniform() # Uniform between -1 and 1 only (at the moment)
Normal(mean=0, std=1)
Zero() # Zero initialized array for biases
```

### Layer Properties
Learning rates can be set at both a network level (every layer) or at individual layers themselves. This is done through the use of a layer properties class. Each layer with trainable parameters has a default learning rate, weight / bias initializer, and optimizer. So even if you input no layer properties for the layer (or network) it will be populated with some defaults. 

```python
# This example would set these specific layer properties for the first and second dense layer
layer1_properties = LayerProperties(learning_rate=0.05, weight_initializer=Uniform(), bias_initializer=Uniform(), optimizer=SGD())
layer2_properties = LayerProperties(learning_rate=0.03, weight_initializer=Uniform(), bias_initializer=Zero(), optimizer=SGD())
network_layers = [
    Dense(28 * 28, 70, layer_properties=layer1_properties),
    Sigmoid(),
    Dense(70, 35, layer_properties=layer2_properties),
    Sigmoid(),
    Dense(35, 10),
    Softmax()
]


# Optionally you can set the layer properties for every layer in the network.
# This is done by setting layer properties on the network class itself.
all_layer_properties = LayerProperties(learning_rate=0.05, weight_initializer=Uniform(), bias_initializer=Uniform(), optimizer=SGD())
network = Network(network_layers, TrainingSet(input_train, output_train, input_test, output_test), \
    loss_function, loss_function_prime, epochs=10, batch_size=1, layer_properties=all_layer_properties)


# You can also choose to overwrite only some properties at both the network and layer level.
# For example the following would only change the learning rate (for all layers) but leave all other defaults the same.
all_layer_properties = LayerProperties(learning_rate=0.05)
network = Network(network_layers, TrainingSet(input_train, output_train, input_test, output_test), \
    loss_function, loss_function_prime, epochs=10, batch_size=1, layer_properties=all_layer_properties)
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
- Implement Layer Input Size Calculations