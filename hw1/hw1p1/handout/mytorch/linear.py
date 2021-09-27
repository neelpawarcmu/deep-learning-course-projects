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
        self.dW = np.zeros(None)
        self.db = np.zeros(None)

        self.momentum_W = np.zeros(None)
        self.momentum_b = np.zeros(None)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x = x
        self.z = np.dot(self.x, self.W) + self.b
        return self.z

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        raise NotImplemented


#delete here on
from time import time 
def main():
    def bias_fn(x):
        return x
    def weight_fn(x):
        return x
    
    X = np.array([[4,3]])
    W = np.array([[4,2,-2],[5,4,5]])
    B = np.array([[1,2,3]])

    print('x: ', X)
    print('W: ', W)
    print('b: ', B)

    z = np.dot(X,W) + B
    t0 = time()
    print('z: ', z)
    t1 = time()
    print((t1-t0)*10000)


main()



    