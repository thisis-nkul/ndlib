#import ndlib
import numpy as np

class MSEloss():
    def __init__(self, Y_hat, Y, batch_size):
        self.m = batch_size
        self.Y = Y
        self.Y_hat = Y_hat
        self.cost = None

    def compute_cost(self):
        self.cost = np.sum((self.Y - self.Y_hat)**2)
        self.cost = (0.5/self.m)*self.cost

        #return self.cost

    def backward(self):
        #return dA of last layer
        return (1/self.m)*(self.Y_hat - self.Y)
