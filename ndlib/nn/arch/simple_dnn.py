#simple Deep Neural Network

import ndlib.nn as nn


class SimpleNN:
    def __init__(self, layer_num, layer_units=[1, 1], initializations = [], activations = []):
        # layer_num: total number of layers excluding input layer
        # layer_units[i]: number of hidden units in layer i input layer is layer0
        # initializations: define initialization, 0 in array means default, i.e., HE initialization
        # activations: activations[i] is activation of (i+1)th layer, if an element is 0: no activation                          applied


        self.layer_num = layer_num
        self.layer_units = layer_units
        self.initializations = initializations
        self.activations = activations
        self.output = None

        self.layers = []                #Collection of layers for our NN
        self.layers.append(nn.InputHolder(0))         #layers[0] is input layer

        #some default inits for activations and weight_initialization
        if len(self.initializations) == 0:
            for i in range(layer_num+1):
                self.initializations.append(0)

        #TODO: implement activations cheching and replace activation[i]=0 with 'linear' or relu
        '''
        if len(self.activations) == 0:
            for i in range(layer_num):
                self.activations.append('identity')

        for j in range(len(self.activations)):
            if self.activations[j] == 0:
                self.activations[j] = 'identity'
        '''

        #checks
        if len(self.layer_units) != self.layer_num+1:
            raise Exception('Hidden Units are not correctly specified for each layer!')

        if  len(activations) != layer_num:
            raise Exception('Activations are not properly defined for layers')


        #creting layers
        for i in range(layer_num):
            if(initializations[i]!=None and initializations[i]!=0):
                self.layers.append(nn.Linear(layer_units[i],
                                        layer_units[i+1],
                                        initialization=initializations[i], activation=activations[i]))

            else:
                self.layers.append(nn.Linear(layer_units[i],
                                        layer_units[i+1], activation=activations[i]))

    def forward(self, inp):
        #inp: input vector/matrix X
        self.layers[0].Z = inp               #totally optional. If this consumes more memory than directly                                             #using input, remove this and use inp directly
                                           #but this might help in backprop, idk. No ig it won't.

        #self.output = inp                   #initialized this way bcz the uppr init might take some more
                                            #memory

        for j in range(self.layer_num):
            self.output = self.layers[j+1](self.layers[j].Z)

        return self.output

    __call__ = forward
