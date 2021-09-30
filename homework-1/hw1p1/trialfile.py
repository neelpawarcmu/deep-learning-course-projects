import numpy as np
#linear layer calcs
input_size = 10
hiddens = [3,4]
output_size = 2

layer_nodes = []
layer_nodes.append(input_size)
layer_nodes.extend(hiddens)
layer_nodes.append(output_size)

linearLayers = [(layer_nodes[layer], layer_nodes[layer+1]) for layer in range(0,len(layer_nodes)-1)]


x = np.linspace(-2001, 2001, 15).reshape((3,5)) 


a = np.max(x, axis=1, keepdims=True)
exponent = np.exp(x-a)
sum1 = np.sum(exponent, axis=1, keepdims=True)
sum2 = np.sum(exponent, axis=1)

print(exponent.shape)
#print(sum2.shape)
#print(x)
'''
x = x - np.max(x)
print(x)'''
#print(np.exp(x))
#print(a.shape)

import numpy as np
x = np.array([1, 2])
W = np.array([[4, 2, -2], [5, 4, 5]])
b = np.array([1,1,1])
delta = np.array([1,-1,1])
zeros = np.zeros((1, 10))
ones = np.ones((1, 10))
print(np.size(zeros, 0), np.size(zeros, 1))

def testDot(x,y):
    return np.dot(x,y)


def backward(x, delta):
    #dx = delta * W # dloss/dz * w
    #dW = np.dot(x.T, delta) # x.T, dloss/dz
    #db = delta # dloss/dz
    x = x[:,None]
    delta = delta[:,None]
    dW = np.dot(x, delta.T) # same shape as W
    db = np.mean(delta, axis=0, keepdims=True)
    dx = np.dot(delta.T, W.T)
    print('dx = ', dx)
    print('dW = ', dW)
    print('db = ', db)
    print('delta = ', delta)
    print('x shape = ', x.shape)
    print('W shape = ', W.shape)
    print('b shape = ', b.shape)
    print('dx shape = ', dx.shape)
    print('dW shape = ', dW.shape)
    print('db shape = ', db.shape)
    return dx
    

