# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        #pseudocode in handout and lecture 1 (or 2) slides @remove
        batch_size, in_channel, input_size = x.shape
        output_size = ((input_size - self.kernel_size) // self.stride) + 1

        #store x and its input dimension to use later in backprop gradient calculations #@change
        self.x = x
        self.input_size = input_size #@change

        Z = np.zeros((batch_size, self.out_channel, output_size))


        #iterate over batches
        for i in range(batch_size):
            batch =  x[i,:,:] # shape => (in_channel, input_size)
            
            #iterate over channels
            for j in range(self.out_channel):
                W = self.W[j,:,:] # shape => (in_channel, input_size)
                b = self.b[j] # shape => (1) scalar

                #iterate over image width 
                for k in range(output_size): # (iterating over output size ensures we dont run out of image for edge cases where filter is longer)
                    start, end = k * self.stride, k * self.stride + self.kernel_size  # indexing for kernel placement over image
                    segment = batch[:, start:end] # shape => (in_channel, kernel_size)
                    affineCombination = np.sum(segment * W) + b # shape => (1) ie. scalar
                    Z[i,j,k] = affineCombination
        return Z

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size, out_channel, output_size = delta.shape

        dx = np.zeros((batch_size, self.in_channel, self.input_size))
        for i in range(batch_size):
            batch =  self.x[i,:,:] # shape => (in_channel, input_size)
            for j in range(self.out_channel):
                W = self.W[j,:,:] # shape => (in_channel, kernel_size)
                b = self.b[j] # shape => (), ie. scalar
                for k in range(output_size):
                    start, end = k * self.stride, k * self.stride + self.kernel_size
                    segment = batch[:, start:end] # shape => (in_channel, kernel_size)
                    delta_local = delta[i,j,k] # shape => (1) ie. scalar

                    # (make sure gradient dimensions match) -> 
                    # dx[i,:,start:end]:(in_channel, kernel_size) 
                    # dW[j,:,:]:(in_channel, kernel_size)
                    # db[j]: (1) ie. scalar

                    dx[i,:,start:end] += W * delta_local
                    self.dW[j,:,:] += segment * delta_local
                    self.db[j] += delta_local
        return dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        
        batch_size, in_channel, input_width, input_height = x.shape
        output_width = ((input_width - self.kernel_size) // self.stride) + 1
        output_height = ((input_height - self.kernel_size) // self.stride) + 1

        #store x and its input dimension to use later in backprop gradient calculations
        self.x = x
        self.input_width = input_width #@change
        self.input_height = input_height #@change

        Z = np.zeros((batch_size, self.out_channel, output_width, output_height))

        for i in range(batch_size):
            batch =  x[i,:,:,:] # shape => (in_channel, input_width, input_height)
            for j in range(self.out_channel):
                W = self.W[j,:,:,:] # shape => (in_channel, kernel_size, kernel_size)
                b = self.b[j] # shape => (1), ie. scalar
                for k in range(output_width):
                    startWidth, endWidth = k * self.stride, k * self.stride + self.kernel_size
                    for l in range(output_height):
                        startHeight, endHeight = l * self.stride, l * self.stride + self.kernel_size
                        segment = batch[:, startWidth:endWidth, startHeight:endHeight] # shape => (in_channel, kernel_size, kernel_size)
                        affineCombination = np.sum(segment * W) + b # shape => (1), ie. scalar
                        Z[i,j,k,l] = affineCombination
        return Z

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, out_channel, output_width, output_height = delta.shape

        dx = np.zeros((batch_size, self.in_channel, self.input_width, self.input_height)) #@change

        for i in range(batch_size):
            batch =  self.x[i,:,:,:] # shape => (in_channel, input_width, input_height)
            for j in range(self.out_channel):
                W = self.W[j,:,:,:] # shape => (in_channel, kernel_size, kernel_size)
                b = self.b[j] # shape => (), ie. scalar
                for k in range(output_width):
                    startWidth, endWidth = k * self.stride, k * self.stride + self.kernel_size
                    for l in range(output_height):
                        startHeight, endHeight = l * self.stride, l * self.stride + self.kernel_size
                        segment = batch[:, startWidth:endWidth, startHeight:endHeight] # shape => (in_channel, kernel_size, kernel_size)
                        delta_local = delta[i,j,k,l] # shape => (1) ie. scalar

                        # (make sure gradient dimensions match) -> 
                        # dx[i,:,start:end]:(in_channel, kernel_size) 
                        # dW[j,:,:]:(in_channel, kernel_size)
                        # db[j]: (1) ie. scalar

                        dx[i,:,startWidth:endWidth,startHeight:endHeight] += W * delta_local
                        self.dW[j,:,:,:] += segment * delta_local
                        self.db[j] += delta_local
        return dx
        

# dilation => 
class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        self.kernel_dilated = (kernel_size-1) * (dilation-1) + kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """

        #save x for backprop? @change

        batch_size, in_channel, input_width, input_height = x.shape

        input_width_padded = input_width + 2 * self.padding
        input_height_padded = input_height + 2 * self.padding

        x_padded = np.zeros((batch_size, in_channel, input_width_padded, input_height_padded))

        # TODO: padding x with self.padding parameter (HINT: use np.pad())
        for b in range(batch_size):
            batch = x[b,:,:,:] 
            for c in range(in_channel): # shape => (in_channel, input_width, input_height)
                x_padded[b,c,:,:] = np.pad(batch[c,:,:], ((self.padding,self.padding),(self.padding,self.padding)), mode='constant', constant_values=0)
                
        # TODO: do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        #       HINT: for loop to get self.W_dilated

        self.W_dilated = np.zeros((self.out_channel, in_channel, self.kernel_dilated, self.kernel_dilated))
        for ch_o in range(self.out_channel):
            for ch_i in range(self.in_channel):
                self.W_dilated[ch_o, ch_i, :, :][::self.dilation,::self.dilation] = self.W[ch_o, ch_i, :, :]


        # TODO: regular forward, just like Conv2d().forward()
        #store x to use later in backprop gradient calculations
        self.x = x
        self.x_padded = x_padded

        #output_size calculations
        output_width = (input_width_padded - self.kernel_dilated) // self.stride
        output_height = (input_height_padded - self.kernel_dilated) // self.stride

        Z = np.zeros((batch_size, self.out_channel, output_width, output_height))
        '''
        x_padded = (batch_size, in_channel, input_width_padded, input_height_padded)
        W_dilated = (out_channel, in_channel, kernel_dilated, kernel_dilated)
        b = (out_channel)
        Z = (batch_size, out_channel, output_width, output_height)
        '''

        for i in range(batch_size):
            batch =  x_padded[i,:,:,:] # shape => (in_channel, input_width_padded, input_height_padded)
            for j in range(self.out_channel):
                W = self.W_dilated[j,:,:,:] # shape => (in_channel, kernel_dilated, kernel_dilated)
                b = self.b[j] # shape => (1), ie. scalar
                for k in range(output_width):
                    startWidth, endWidth = k * self.stride, k * self.stride + self.kernel_dilated
                    for l in range(output_height):
                        startHeight, endHeight = l * self.stride, l * self.stride + self.kernel_dilated
                        segment = batch[:, startWidth:endWidth, startHeight:endHeight] # shape => (in_channel, kernel_dilated, kernel_dilated)
                        affineCombination = np.sum(segment * W) + b # shape => (1), ie. scalar
                        Z[i,j,k,l] = affineCombination
        return Z


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # TODO: main part is like Conv2d().backward(). The only difference are: we get padded input and dilated kernel
        #       for whole process while we only need original part of input and kernel for backpropagation.
        #       Please refer to writeup for more details.
        
        batch_size, out_channel, output_width, output_height = delta.shape

        output_width_padded = output_width + 2 * self.padding
        output_height_padded = output_height + 2 * self.padding

        delta_dilated = np.zeros((batch_size, out_channel, output_width_padded, output_height))
        for ch_o in range(self.out_channel):
            for ch_i in range(self.in_channel):
                self.W_dilated[ch_o, ch_i, :, :][::self.dilation,::self.dilation] = self.W[ch_o, ch_i, :, :]



        dx = np.zeros_like(self.x) # shape => (batch_size, in_channel, input_width, input_height)
        for i in range(batch_size):
            batch =  self.x[i,:,:,:] # shape => (in_channel, input_width, input_height)
            for j in range(self.out_channel):
                W = self.W[j,:,:,:] # shape => (in_channel, kernel_size, kernel_size)
                for k in range(output_width):
                    startWidth, endWidth = k * self.stride, k * self.stride + self.kernel_size
                    for l in range(output_height):
                        startHeight, endHeight = l * self.stride, l * self.stride + self.kernel_size
                        segment = batch[:, startWidth:endWidth, startHeight:endHeight] # shape => (in_channel, kernel_size, kernel_size)
                        delta_local = delta[i,j,k,l] # shape => (1) ie. scalar

                        # (make sure gradient dimensions match) -> 
                        # dx[i,:,start:end]:(in_channel, kernel_size) 
                        # dW[j,:,:]:(in_channel, kernel_size)
                        # db[j]: (1) ie. scalar

                        dx[i,:,startWidth:endWidth,startHeight:endHeight] += W * delta_local
                        self.dW[j,:,:,:] += segment * delta_local
                        self.db[j] += delta_local
        return dx



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        dx = np.reshape(x, (self.b, self.c * self.w))
        return dx

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        dx = np.reshape(delta, (self.b, self.c, self.w))
        return dx