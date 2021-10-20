# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros((in_feature, out_feature), dtype=float)
        self.db = np.zeros((1, out_feature), dtype=float)

        self.momentum_W = np.zeros((in_feature, out_feature), dtype=float)
        self.momentum_b = np.zeros((1, out_feature), dtype=float)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        #delta = dloss/doutput
        #z = W * x + b           #1, out  =>  (in, out) * (batch, in) + (1, out)         # sum(x * W) + b
        self.x = x
        self.z = (np.dot(self.x, self.W)) + self.b
        return self.z

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        #delta = dloss/doutput
        #dx = delta * W          #batch, in  =>  (batch, out) * (in, out).T         # delta * W.T
        #dW = x * delta          #in, out    =>  (batch, in) * (batch, out)         # x.T * delta
        #db = delta              #1, out     =>  (batch, out)                       # sum(delta, axis = 0, keepdims=True)
        
        batch_dx = np.dot(delta, self.W.T)
        batch_dW = np.dot(self.x.T, delta)
        batch_db = np.sum(delta, axis=0, keepdims=True)

        #compute averages as gradients are the batch average of loss wrt batch parameters
        batch_size = delta.shape[0]
        self.dW, self.db = batch_dW/batch_size, batch_db/batch_size

        return batch_dx