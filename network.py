from config import *
if enableCuda:
    print("Cuda Enabled.")
    import cupy as np
else:
    print("Cuda Disabled.")
    import numpy as np
import time

class Network():

    # The total training time in minutes.
    totalTrainingTime = 0

    def __init__(self, layers, loss, loss_prime, x_train, y_train, x_test, y_test, epochs = 1000, learning_rate = 0.01, verbose = True):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self):
        if self.verbose:
            print("Beginning training...")
            startTime = time.time()

        for e in range(self.epochs):
            trainingError = 0
            for x, y in zip(self.x_train, self.y_train):

                # Forward Propagation
                output = self.predict(x)

                # Error Calculation (For Debug Only)
                trainingError += self.loss(y, output)

                # Backward Propagation
                grad = self.loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, self.learning_rate)

            trainingError /= len(self.x_train)
            if self.verbose:
                ratioIncorrect = self.test()
                #Calculate estimated training time remaining for my sanity
                endTime = time.time()
                timeElapsedMins = (endTime - startTime) / 60
                timePerEpoch = timeElapsedMins / (e+1)
                epochsRemaining = self.epochs - (e+1)
                trainingTimeRemaining = timePerEpoch * epochsRemaining
                print("{}/{}, network training error = {:.4f}, test percentage incorrect = {:.2%}, training time remaining = {:.2f} minutes".format((e+1), self.epochs, trainingError, ratioIncorrect, trainingTimeRemaining))
        
        endTime = time.time()
        timeElapsedMins = (endTime - startTime) / 60
        self.totalTrainingTime += timeElapsedMins

        if self.verbose:
            print("Training Complete. Elapsed Time = {:.2f} seconds. Or {:.2f} minutes.".format(endTime - startTime, timeElapsedMins))

    # Returns the ratio of incorrect responses in the training set
    def test(self):
        numCorrect = 0
        numIncorrect = 0
        for x, y in zip(self.x_test, self.y_test):
            output = self.predict(x)
            if np.argmax(output) == np.argmax(y):
                numCorrect+=1
            else:
                numIncorrect+=1
        return numIncorrect / (numCorrect + numIncorrect)