

class Loss:
    '''

        BASE CLASS FOR ALL LOSS FUNCTIONS

    '''
    def __init__(self, Y_hat, Y, batch_size):
        self.m = batch_size
        self.Y = Y
        self.Y_hat = Y_hat
        self.cost = None

    def compute_cost(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
