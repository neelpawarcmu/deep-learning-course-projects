# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        # handle very big and small exponents using log - sum - exponent trick
        # TO-DO: can mean be better for a?
        # a = np.mean(x, axis=1, keepdims=True)
        a = np.max(x, axis=1, keepdims=True)
        exponents = np.exp(x-a)
        sum = np.sum(exponents, axis=1, keepdims=True)
        
        softmax_predictions = exponents / sum

        logLosses = - self.labels * np.log(softmax_predictions)
        crossEntropyLoss = np.sum(logLosses, axis=1)

        self.softmax_predictions = softmax_predictions
        return crossEntropyLoss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return self.softmax_predictions - self.labels


#delete here on
#how to test using inputs cause => hidden input and output not specified. pytest?
#red comments 
#where are we implementing the complete MLP
#is this how tf does things when we design using it? split layers?