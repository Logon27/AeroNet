class TrainingSet():

    def __init__(self, input_train, output_train, input_test, output_test, post_processing = lambda x : x):
        # Datasets (numpy arrays)
        # Training Set
        self._input_train = input_train
        self._output_train = output_train
        # Validation Set
        self._input_test = input_test
        self._output_test = output_test

        # Applied to the network prediction and training data before equivalency comparison.
        # By default it applies no post processing
        self._post_processing = post_processing
        
        # Calculate dataset sizes
        self._input_train_size = len(self.input_train)
        self._output_train_size = len(self.output_train)
        self._input_test_size = len(self.input_test)
        self._output_test_size = len(self.output_test)
    
    @property
    def input_train(self):
        return self._input_train
    
    @property
    def output_train(self):
        return self._output_train

    @property
    def input_test(self):
        return self._input_test
    
    @property
    def output_test(self):
        return self._output_test
    
    @property
    def post_processing(self):
        return self._post_processing

    @property
    def input_train_size(self):
        return self._input_train_size
    
    @property
    def output_train_size(self):
        return self._output_train_size

    @property
    def input_test_size(self):
        return self._input_test_size
    
    @property
    def output_test_size(self):
        return self._output_test_size

    def __str__(self):
        training_str = "{:<15} {} {}".format("Training Data:", self.input_train_size, "samples")
        test_str = "{:<15} {} {}".format("Test Data:", self.input_test_size, "samples")
        return training_str + '\n' + test_str