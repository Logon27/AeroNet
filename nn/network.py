import numpy as np
import time

class Network():

    # The total training time in minutes.
    totalTrainingTime = 0

    def __init__(self, layers, loss, loss_prime, x_train, y_train, x_test, y_test, epochs = 1000, learning_rate = 0.01, batch_size = 1, verbose = True):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose

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
            for batch in self.iterate_minibatches(self.x_train, self.y_train, self.batch_size, shuffle=True):
                # Unpack batch training data
                x_batch, y_batch = batch
                # Track all gradients for the batch within a list
                gradients = []

                # Calculate the gradient for all training samples in the batch
                for x, y in zip(x_batch, y_batch):
                    # Forward Propagation
                    output = self.predict(x)

                    # Calculate Gradient
                    gradients.append(self.loss_prime(y, output))
                    
                # Average all the gradients calculated in the batch
                gradient = np.mean(gradients, axis=0)

                # Backward Propagation
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learning_rate)

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
        for x, y in zip(self.x_train, self.y_train):
            output = self.predict(x)
            if np.argmax(output) == np.argmax(y):
                numCorrect += 1
            else:
                numIncorrect += 1
        accuracyTrain = numCorrect / (numCorrect + numIncorrect)

        # Test Accuracy
        numCorrect = 0
        numIncorrect = 0
        for x, y in zip(self.x_test, self.y_test):
            output = self.predict(x)
            if np.argmax(output) == np.argmax(y):
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

        print("{:<15} {} {}".format("Training Data:", len(self.x_train), "samples"))
        print("{:<15} {} {}".format("Test Data:", len(self.x_test), "samples"))
        print("{:<15} {}".format("Loss Function:", self.loss.__name__))
        print("{:<15} {}".format("Epochs:", str(self.epochs)))
        print("{:<15} {}".format("Learning Rate:", str(self.learning_rate)))
        print("{:<15} {}".format("Batch Size:", str(self.batch_size)))
        print("{:<15} {}".format("Verbose:", self.verbose))
        print("\n===== End Network Information =====\n")
