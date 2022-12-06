import numpy as np
import time
from nn.network.training_set import TrainingSet
from nn.layers.layer_properties import LayerProperties

from copy import deepcopy

class Network():

    # The total training time in minutes.
    totalTrainingTime = 0

    def __init__(self, layers, training_set: TrainingSet, loss, loss_prime, epochs = 1000, batch_size = 1, verbose = True, layer_properties: LayerProperties = None):
        self.layers = layers
        self.training_set = training_set
        self.loss = loss
        self.loss_prime = loss_prime
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        # Optionally set the layer properties for all layers that utilize layer properties parameters
        if layer_properties is not None:
            self.layer_properties = layer_properties
            for layer in self.layers:
                if hasattr(layer, 'layer_properties'):
                    # Replace all layer defaults with any non "None" layer properties.
                    # This is just a lot of fancy code to allow you to override only 'some' of the default layer properties.
                    # Instead of forcing you to populate all the parameters every time.
                    for attr, _ in layer.layer_properties.__dict__.items():
                        if getattr(layer_properties, attr) is not None:
                            # copy is necessary to ensure that individual layer classes don't get shared instances of an optimizer
                            # optimizers such as momentum sgd require separate instances to track velocity
                            setattr(layer.layer_properties, attr, deepcopy(getattr(layer_properties, attr)))

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self):
        if self.verbose:
            self.printNetworkInfo()
            print("Beginning training...")
            startTime = time.time()

        for epoch in range(self.epochs):

            #Optimization method            Samples in each gradient calculation        Weight updates per epoch
            #Batch Gradient Descent         The entire dataset	                        1
            #Minibatch Gradient Descent	    Consecutive subsets of the dataset	        n / size of minibatch
            #Stochastic Gradient Descent	Each sample of the dataset	                n
            #Increasing the batch size increases the number of epoches required for convergence
            for batch in self.iterate_minibatches(self.training_set.input_train, self.training_set.output_train, self.batch_size, shuffle=True):
                # Unpack batch training data
                input_train_batch, output_train_batch = batch
                # Track all gradients for the batch within a list
                gradients = []

                # Calculate the gradient for all training samples in the batch
                for input_train_sample, output_train_sample in zip(input_train_batch, output_train_batch):
                    # Forward Propagation
                    prediction = self.predict(input_train_sample)

                    # Calculate Gradient
                    gradients.append(self.loss_prime(output_train_sample, prediction))
                    
                # Average all the gradients calculated in the batch
                gradient = np.mean(gradients, axis=0)

                # Backward Propagation
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)

            if self.verbose:
                accuracyTrain, accuracyTest = self.test()
                #Calculate estimated training time remaining for my sanity
                endTime = time.time()
                timeElapsedMins = (endTime - startTime) / 60
                timePerEpoch = timeElapsedMins / (epoch+1)
                epochsRemaining = self.epochs - (epoch+1)
                trainingTimeRemaining = timePerEpoch * epochsRemaining
                print("{}/{}, Accuracy Train = {:.2%}, Accuracy Test = {:.2%}, Time Remaining = {:.2f} minutes".format((epoch+1), self.epochs, accuracyTrain, accuracyTest, trainingTimeRemaining))
        
        endTime = time.time()
        timeElapsedMins = (endTime - startTime) / 60
        self.totalTrainingTime += timeElapsedMins

        if self.verbose:
            print("Training Complete. Elapsed Time = {:.2f} seconds. Or {:.2f} minutes.".format(endTime - startTime, timeElapsedMins))

    # Returns the accuracy against the training and test datasets
    def test(self):
        # Training Accuracy
        numCorrect = 0
        numIncorrect = 0
        for input_train_sample, output_train_sample in zip(self.training_set.input_train, self.training_set.output_train):
            prediction = self.predict(input_train_sample)
            if self.training_set.post_processing(prediction) == self.training_set.post_processing(output_train_sample):
                numCorrect += 1
            else:
                numIncorrect += 1
        accuracyTrain = numCorrect / (numCorrect + numIncorrect)

        # Test Accuracy
        numCorrect = 0
        numIncorrect = 0
        for input_train_sample, output_train_sample in zip(self.training_set.input_test, self.training_set.output_test):
            prediction = self.predict(input_train_sample)
            if self.training_set.post_processing(prediction) == self.training_set.post_processing(output_train_sample):
                numCorrect += 1
            else:
                numIncorrect += 1
        accuracyTest = numCorrect / (numCorrect + numIncorrect)

        return accuracyTrain, accuracyTest
    
    # Source: https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    # You should ideally shuffle the data. Take XOR for example if you have a batch size of 2.
    # And your batch pairs [0, 0] = [0] and [0, 1] = [1] it will average the gradient of these two examples every epoch.
    # Which means you will almost never reach a solution.
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0], batchsize):
            end_idx = min(start_idx + batchsize, inputs.shape[0])
            if shuffle:
                excerpt = indices[start_idx:end_idx]
            else:
                excerpt = slice(start_idx, end_idx)
            yield inputs[excerpt], targets[excerpt]

    def printNetworkInfo(self):

        print("===== Network Information =====")
        print("Network Architecture:")
        print("[")
        print(*self.layers, sep='\n')
        print("]\n")

        print("{:<15} {} {}".format("Training Data:", self.training_set.input_train_size, "samples"))
        print("{:<15} {} {}".format("Test Data:", self.training_set.input_test_size, "samples"))
        print("{:<15} {}".format("Loss Function:", self.loss.__name__))
        print("{:<15} {}".format("Epochs:", str(self.epochs)))

        if hasattr(self, 'layer_properties'):
            print("{:<15} {}".format("Learning Rate:", str(self.layer_properties.learning_rate)))
            print("{:<15} {}".format("Optimizer:", str(self.layer_properties._weight_optimizer.__class__.__name__)))

        print("{:<15} {}".format("Batch Size:", str(self.batch_size)))
        print("{:<15} {}".format("Verbose:", self.verbose))
        print("\n===== End Network Information =====\n")
