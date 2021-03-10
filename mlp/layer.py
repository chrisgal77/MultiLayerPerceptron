import numpy as np 
from functions import ACTIVATIONS, ACT_DERIVATIVES
from preprocessing import OneHotEncoder


class Layer:
    def __init__(self, n_neuron, n_input, lr, activation = 'sigmoid', l2 = 0):
        self.n_neuron = n_neuron
        self.activation = ACTIVATIONS[activation]
        self.act_derivative = ACT_DERIVATIVES[activation]
        self.weight = np.random.normal(loc=0.0, scale=0.1, size=(n_input, self.n_neuron))
        self.bias = np.zeros(shape=(n_neuron))
        self.l2 = l2
        self.lr = lr

    def forward(self, X):
        
        self.z = np.dot(X, self.weight) + self.bias
        self.a = self.activation(self.z)

        return self.z, self.a

    def backward(self, delta, previous_a, previous_z = None):
        try:
            sigma_bias = np.sum(delta, axis=0)
            sigma_weight= np.dot(previous_a.T, delta) + self.l2 * self.weight

            delta = np.dot(delta, self.weight.T) * self.act_derivative(previous_z)

            self.weight -= self.lr * sigma_weight 
            self.bias -= self.lr * sigma_bias

            return delta

        except:

            sigma_bias = np.sum(delta, axis=0)
            sigma_weight= np.dot(previous_a.T, delta) + self.l2 * self.weight

            self.weight -= self.lr * sigma_weight 
            self.bias -= self.lr * sigma_bias
        
            return None


if __name__ == '__main__':

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    n_samples, n_featues = X.shape

    lay1 = Layer(30, n_featues, lr=0.01)
    lay2 = Layer(10, 30, lr=0.01)
    lay3 = Layer(3, 10, lr= 0.01)

    y = OneHotEncoder().encode(y)

    def cost_der(y_true, output):
        return (output - y_true) 

    for _ in range(50):
        z1, a1 = lay1.forward(X)
        z2, a2 = lay2.forward(a1)
        z3, a3 = lay3.forward(a2)

        print( 1/2 * np.sum(np.square(a3 - y)) )

        delta = cost_der(y, a3) * ACTIVATIONS['sigmoid'](z3)

        delta = lay3.backward(delta, a2, z2)
        delta = lay2.backward(delta, a1, z1)

        lay1.backward(delta, X)

    


