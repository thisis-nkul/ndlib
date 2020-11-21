import numpy as np

import ndlib.functional as F




class Linear:

    #TODO: implement activations within the layer as to make backprop implem. easier
    #and THINK: is it really necessary to keep self.A?

    def __init__(self, input_dim, output_dim, W=0, b=0, activation = 'identity', initialization='HE'):

        super(Linear, self).__init__()
        if W == 0:
            W = getattr(F, initialization + '_initialization')(input_dim, output_dim)

        if b == 0:
            b = np.random.randn(output_dim, 1)

        self.W = W
        self.dW = None
        self.db = None
        self.dZ = None
        self.dA = None
        self.b = b
        self.activation = activation
        self.input = None
        self.A = 0
        self.Z = 0

    def forward(self, X):
        self.input = X          #for backprop

        #for backprop

        self.Z = np.dot(self.W, X) + self.b    #for backprop
        self.A = getattr(F, self.activation)(self.Z)
        return self.A


        return self.A

    __call__ = forward

########### UNDER DEVELopment
    def backward(self, dA, batch_size):
        '''DOESN'T WORK WITH SOFTMAX YET.'''
    #   intuition: self.activation.prime(dA_prev_layer, self.Z, self.A)
    #   abv code might be wrong but you get the idea, right?
    #   dA = None for last layer, loss
    #   for last layer optimizer/loss_function will provide dZ directly
        m = batch_size
        ##work on below section is much needed
        if self.activation != 'softmax':
            self.dA = dA
            self.dZ = self.dA*getattr(F, self.activation + '_prime')(self.Z)

        elif self.activation == 'softmax':
            #assumes softmax is last layer only! use carefully. I'm still learning :P
            self.dZ = dA                   #A lil' hack
        #####


        self.dW = (1/m)*np.dot(self.dZ, self.input.T) #will self.input.T will work for 3-d tensor also?
        self.db = (1/m)*np.sum(self.dZ, axis=1, keepdims=True)

        return np.dot(self.W.T, self.dZ)        #dA[l-1] or dA_prev_layer

###########################################
    def parameters(self):
        return {'weights': self.W, 'bias': self.b}

    def parameters_shape(self):
        return {'W': self.W.shape, 'b': self.b.shape}




class InputHolder():
    def __init__(self, data):
        self.Z = data      #for easier implementation of Simple NN

    def backward(self, dA, batch_size):
        pass
