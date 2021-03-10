import numpy as np 
from functions import ACTIVATIONS, ACT_DERIVATIVES

class Layer:
    def __init__(self, n_neuron, n_input, activation):
        self.n_neuron = n_neuron
        self.activation = ACTIVATIONS[activation]
        self.act_derivative = ACT_DERIVATIVES[activation]
        self.weight = np.random.normal(loc=0.0, scale=0.1, size=(n_input, self.n_neuron))
        self.bias = np.zeros(shape=(n_neuron))

    def forward(self, X):
        
        self.z = np.dot(X, self.weight) + self.bias
        self.a = self.activation(self.z)

        return self.z, self.a

    def backward(self, delta):
        pass
