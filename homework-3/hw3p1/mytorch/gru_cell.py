import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        #IDL lec15 slide22 + hw3p1 writeup pg 9 @remove

        #r_t
        term1 = np.dot(self.Wrx, x) + self.bir # (h, d) * (d,) = (h,)
        term2 = np.dot(self.Wrh, h) + self.bhr # (h, h) * (h,) = (h,)
        r_t = self.r_act(term1 + term2)         # (h,)
        self.r = r_t #store value

        #z_t
        term1 = np.dot(self.Wzx, x) + self.biz # (h, d) * (d,) = (h,)
        term2 = np.dot(self.Wzh, h) + self.bhz # (h, h) * (h,) = (h,)
        z_t = self.z_act(term1 + term2)         # (h,)
        self.z = z_t #store value

        #n_t 
        term1 = np.dot(self.Wnx, x) + self.bin # (h, d) * (d,) = (h,)
        term2 = np.dot(self.Wnh, h) + self.bhn # (h, h) * (h,) = (h,)
        term2 *= r_t                           # (h,) x (h,) = (h,) elem-wise multiplication
        n_t = self.h_act(term1 + term2)         
        self.n = n_t #store value              #(h,)

        #h_t
        term1 = (1-z_t) * n_t 
        term2 = z_t * h
        h_t = term1 + term2

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t


    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim) #### this is basically dh_t, derivative of an h_t from forward
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly


        input_dim, = self.x.shape
        hidden_dim, = self.hidden.shape

        #input = 5, hidden = 2
        #derivatives are row vectors and actuals are column vectors. 
        #to begin with, delta shape is good to go as given by them
        #some changes to pdf for gru but the small subparts only, overall it is the same
        #calculate derivatives as given in pdf and transpose in the end to get shape of der = shape of actual
        #all things should be vectors like (1,5)
        #just folllow ppt very carefully and you will get around 26 equations for fwd
        #Follow the derivatives from the saved values in these equations

        #
        self.x = self.x.reshape(1,-1)
        self.hidden = self.hidden.reshape(1,-1)
        self.r = self.r.reshape(1,-1)
        self.z = self.z.reshape(1,-1)
        self.n = self.n.reshape(1,-1)

        print(f'x shape: {self.x.shape}')
        print(f'hidden shape: {self.hidden.shape}')
        print(f'r shape: {self.r.shape}')
        print(f'z shape: {self.z.shape}')
        print(f'n shape: {self.n.shape}')

        #layer 1
        dz_t = - self.n + self.hidden            # (1,h) 
        dn_t = 1 - self.z                        # (1,h) 
        dh_t_partA = self.z                      # (1,h)
        print(f'dz_t shape: {dz_t.shape}')
        print(f'dn_t shape: {dn_t.shape}')
        print(f'dh_t_partA shape: {dh_t_partA.shape}')

        #layer 2
        dtanh_n = 1 - self.n**2                  # (1,h) 
        dnt_Wnx = np.dot(self.x.T, dtanh_n).T    # (d,1) * (1,h) = (d,h).T = (h,d)
        self.dWnx = dn_t * dnt_Wnx               # (1,h) * ()              = (h,d)
        print(f'dtanh_n shape: {dtanh_n.shape}') 
        print(f'dnt_Wnx shape: {dnt_Wnx.shape}') 
        print(f'self.dWnx shape: {self.dWnx.shape}') 




        '''self.x = self.x.reshape(input_dim, 1)
        self.hidden = self.hidden.reshape(hidden_dim, 1)
        self.z = self.z.reshape(hidden_dim, 1)
        self.n = self.n.reshape(hidden_dim, 1)

        d0 = delta #d16, d15
        d1 = self.z * d0 # (h,1) * (h,1) =  #d13
        d2 = self.hidden * d0 # (h,1) * (h,)
        d3 = self.n * d0 # (h,1) * (h,)
        d4 = -1 * d3 # (h,)
        d5 = d2 + d4 # (h,)
        d6 = (1-self.z)*d0 # (h,1) * (h,)
        d7 = d5 * self.z * (1-self.z) # (h,)*(h,1)*(h,1)
        d8 = d6 * (1-self.n**2) # (h,)*(h,1)
        d9 = np.dot(d8, self.Wnx.T) #Uh  # (h,)*(h,d) = (???)
        d10 = np.dot(d8, self.Wnh.T) #Wh # ().(h,h)
        d11 = np.dot(d7, self.Wzx.T) #Uz # ().()
        d12 = np.dot(d7, self.Wzh.T) #Wz # ().()
        d14 = d10 * self.r # ()*()
        d15 = d10 * self.hidden # ()*()
        d16 = d15 * self.r * (1-self.r) # ()*()*()
        d13 = np.dot(d16, self.Wrx.T) #Ur # ().()
        d17 = np.dot(d16, self.Wrh.T) #Wr # ().()

        print('delta:', delta.shape)
        print(f'd0: {delta.shape} = {d0.shape}')
        print(f'd1: {self.z.shape} * {d0.shape} = {d1.shape}')
        print(f'd2: {self.hidden.shape} * {d0.shape} = {d2.shape}')
        print(f'd3: {self.n.shape} * {d0.shape} = {d3.shape}')
        print(f'd4: {d3.shape} = {d4.shape}')
        print(f'd5: {d2.shape} + {d4.shape} = {d5.shape}')
        print(f'd6: {self.z.shape} * {d0.shape} = {d6.shape}')
        print(f'd7: {d5.shape} * {self.z.shape} * {self.z.shape} = {d7.shape}')
        print(f'd8: {d6.shape} * {self.n.shape} = {d8.shape}')
        print(f'd9: {d8.shape} dot {self.Wnx.shape} = {d9.shape}')
        print(f'd10: {d8.shape} dot {self.Wnh.T.shape} {d10.shape}')
        print(f'd11: {d11.shape} dot {self.Wzx.shape}')
        print(f'd12: {d7.shape} dot {self.Wzh.shape} = {d12.shape}')
        print(f'd14: {d10.shape} * {self.r.shape} = {d14.shape}')
        print(f'd15: {d10.shape} * {self.hidden.shape} = {d15.shape}')
        print(f'd16: {d15.shape} * {self.r.shape} * {self.r.shape} = {d16.shape}')
        print(f'd13: {d16.shape} dot {self.Wrx.shape} = {d13.shape}')
        print(f'd17: {d17.shape} dot {self.Wrh.T.shape} = {d17.shape}')

        dx = d9 + d11 + d13
        dh = d12 + d14 + d1 + d17
        self.dWrx = np.dot(self.x.T, d16) #dUr
        self.dWzx = np.dot(self.x.T, d7) #dUz
        self.dWnx = np.dot(self.x.T, d8) #dUh
        self.dWrh = np.dot(self.hidden.T, d16) #dWr
        self.dWzh = np.dot(self.hidden.T, d7) #dWz
        self.dWnh = np.dot((self.hidden.T * self.r).T, d8) #dWh

        print('x:', self.x.shape, 'dx:', dx.shape)
        print('h:', self.hidden.shape, 'dh:', dh.shape)'''
