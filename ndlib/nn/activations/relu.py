import ndlib.functional as F

class relu:
    def __init__(self, input):
        self.input = input
        self.forward()

    def forward(self):
        self.output = F.relu(self.input)
        return self.output

    def backward(self):
        pass
