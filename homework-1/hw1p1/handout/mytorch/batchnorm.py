# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        self.x = x

        # if eval:
        # testing: use running means and dont store   
        # # training: compute means and store @@@@@@langclass @@syntaxshuff if eval line return and body add for else
        #norm = (batch, in), gamma = (1, in) => 
        if eval:
            mean = self.running_mean #mean = (1, in)
            var = self.running_var #var = (1, in)
            norm = (x - mean) / (np.sqrt(var + self.eps)) #norm = (batch, in)
            out = (self.gamma * norm) + self.beta #
            self.norm = norm
            self.out = out
            return out
        else:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            norm = (x - mean) / (np.sqrt(var + self.eps))  # 1. normalize to unit size and center
            out = (self.gamma * norm) + self.beta    # 2. affine transformation

        #store as class variables for running mean
        self.mean = mean 
        self.var = var

        #store as class variables for backprop
        self.norm = norm
        self.out = out

        # Update running batch statistics 
        self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        # To do: add comments and improve notations like first term etc. 
        batch_size = delta.shape[0] 
        
        #repititive term saved for ease
        sqrt_var_eps = np.sqrt(self.var + self.eps)

        #update gradients
        self.dgamma = np.sum(delta * self.norm, axis = 0, keepdims=True)
        self.dbeta = np.sum(delta, axis = 0, keepdims=True)
        
        #calculate dnorm and dvar
        gradNorm = self.gamma * delta
        gradVar = -0.5*(np.sum((gradNorm * (self.x - self.mean) / (sqrt_var_eps**3)), axis = 0))

        #calculate dmu
        first_term_dmu = - np.sum(gradNorm/sqrt_var_eps, axis = 0)
        second_term_dmu = - (2/batch_size)*(gradVar)*(np.sum(self.x-self.mean, axis = 0))
        gradMu = first_term_dmu + second_term_dmu

        #calculate dx = f(dnorm) + g(dvar) + h(dmu)
        first_term_dx = gradNorm / sqrt_var_eps
        second_term_dx = gradVar * (2/batch_size) * (self.x-self.mean)
        third_term_dx = gradMu * (1/batch_size)
        dx = first_term_dx + second_term_dx + third_term_dx

        return dx
