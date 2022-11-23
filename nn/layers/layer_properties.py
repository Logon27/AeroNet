from nn.initializers.initializer import Initializer
from nn.optimizers.optimizer import Optimizer

class LayerProperties():

    def __init__(self, learning_rate = None, weight_initializer = None, bias_initializer = None, optimizer = None):
        self._learning_rate = learning_rate
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._optimizer = optimizer
    
    @property
    def learning_rate(self) -> int:
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError("Learning rate must be greater than zero.")
        self._learning_rate = value

    @property
    def weight_initializer(self) -> Initializer:
        return self._weight_initializer
    
    @property
    def bias_initializer(self) -> Initializer:
        return self._bias_initializer

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer
    
    #Need to add setters for the initializer and optimizer and check for the class types