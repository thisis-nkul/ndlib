import numpy as np
import ndlib

class CrossEntropyLoss(ndlib.Loss):

    def __init__(self, Y_hat, Y, batch_size):
        super().__init__(Y_hat, Y, batch_size)

    def compute_cost(self):
        self.cost = (1/self.m)*np.sum(-self.Y*np.log(self.Y_hat))

    def backward(self):
        return self.Y_hat - self.Y
        #returns dZ and not dA
