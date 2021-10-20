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
        # x,exp(x-a) (batch, 10) => softmax (batch, 10) => CELoss (batch,)  # sum(axis=1)
        self.logits = x
        self.labels = y
        # handle very big and small exponents using log - sum - exponent trick
        a = np.max(x, axis=1, keepdims=True)
        # To do: can mean be better for a?
        # a = np.mean(x, axis=1, keepdims=True)

        #softmax = individual exponents / sum of individual exponents
        exponents = np.exp(x-a)
        sum = np.sum(exponents, axis=1, keepdims=True)
        softmax_predictions = exponents / sum

        batchLosses = - self.labels * np.log(softmax_predictions)
        crossEntropyLoss = np.sum(batchLosses, axis=1)

        #store for backprop
        self.softmax_predictions = softmax_predictions
        self.CELoss = crossEntropyLoss
        return self.CELoss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        # softmax (batch, 10) => der_softmax (batch, 10)
        # derivative of softmax wrt its input = softmax - its input
        return self.softmax_predictions - self.labels