import numpy as np

def identity(inp):
    return inp

def relu(inp):
    return np.maximum(inp, 0)

def softmax(inp):
    intermediate = np.exp(inp)

    return np.divide(intermediate, np.sum(intermediate, axis=0))
    #idk if this is robust

#----------------------------------------------------------------------------------------------#
#                                	WEIGHT INITIALIZERS                                        #
#----------------------------------------------------------------------------------------------#
#idk if these initializations will generalize to CNNs, IG they will


def xavier_initialization(input_dim, output_dim):
    return np.random.randn(output_dim, input_dim)*np.sqrt(1/input_dim)

def HE_initialization(input_dim, output_dim):
    return np.random.randn(output_dim, input_dim)*np.sqrt(2/input_dim)



###-------------------------------WEIGHT INITIALIZERS ENDED-----------------------------------#
