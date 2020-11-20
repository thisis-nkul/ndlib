import numpy as np

import ndlib.functional as F




class Linear:

    #TODO: implement activations within the layer as to make backprop implem. easier

    def __init__(self, input_dim, output_dim, W=0, b=0, activation = 'identity', initialization='HE'):

        super(Linear, self).__init__()
        if W == 0:
            W = getattr(F, initialization + '_initialization')(input_dim, output_dim)

        if b == 0:
            b = np.random.randn(output_dim, 1)

        self.W = W
        self.b = b
        self.activation = activation

    def forward(self, X):
        self.input = X          #for backprop

        #for backprop

        self.Z = np.dot(self.W, X) + self.b    #for backprop
        self.A = getattr(F, self.activation)(self.Z)
        return self.A


        return self.A

    __call__ = forward

    #def backward(dA_prev_layer):
    #   self.activation.backward(dA_prev_layer, self.Z, self.A)
    #   abv code might be wrong but you get the idea, right?

    def parameters(self):
        return {'weights': self.W, 'bias': self.b}

    def parameters_shape(self):
        return {'W': self.W.shape, 'b': self.b.shape}




class InputHolder():
    def __init__(self, data):
        self.Z = data      #for easier implementation of Simple NN
