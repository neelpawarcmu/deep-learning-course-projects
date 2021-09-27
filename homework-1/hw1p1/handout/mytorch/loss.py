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
        a = np.mean(x)
        #to do: check this logsumexp part
        x = a + np.log(np.exp(x-a)) #handle very big and small exponents using log - sum - exponent trick
        self.softmaxLoss = np.exp(x) / np.sum(np.exp(x))
        return self.softmaxLoss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        raise NotImplemented


#delete here on
#how to test using inputs cause => hidden input and output not specified. pytest?
#red comments 
#where are we implementing the complete MLP
#is this how tf does things when we design using it? split layers?