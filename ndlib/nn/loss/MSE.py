#import ndlib
import numpy as np

class MSEloss():
    def __init__(self, training_examples, Y, Y_hat):
        self.m = training_examples
        self.Y = Y
        self.Y_hat = Y_hat
        self.cost = None

    def compute(self):
        self.cost = np.sum((self.Y - self.Y_hat)**2)
        self.cost = (0.5/self.m)*self.cost

        return self.cost

    def backward(self):
        pass
