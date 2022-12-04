class TrainingSet():

    def __init__(self, x_train, y_train, x_test, y_test, post_processing = lambda x : x):
        # Datasets (numpy arrays)
        # Training Set
        self._x_train = x_train
        self._y_train = y_train
        # Validation Set
        self._x_test = x_test
        self._y_test = y_test

        # Applied to the network prediction and training data before equivalency comparison.
        # By default it applies no post processing
        self._post_processing = post_processing
        
        # Calculate dataset sizes
        self._x_train_size = len(self.x_train)
        self._y_train_size = len(self.y_train)
        self._x_test_size = len(self.x_test)
        self._y_test_size = len(self.y_test)
    
    @property
    def x_train(self):
        return self._x_train
    
    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test
    
    @property
    def y_test(self):
        return self._y_test
    
    @property
    def post_processing(self):
        return self._post_processing

    @property
    def x_train_size(self):
        return self._x_train_size
    
    @property
    def y_train_size(self):
        return self._y_train_size

    @property
    def x_test_size(self):
        return self._x_test_size
    
    @property
    def y_test_size(self):
        return self._y_test_size

    def __str__(self):
        training_str = "{:<15} {} {}".format("Training Data:", self.x_train_size, "samples")
        test_str = "{:<15} {} {}".format("Test Data:", self.x_test_size, "samples")
        return training_str + '\n' + test_str