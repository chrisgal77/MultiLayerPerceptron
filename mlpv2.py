import numpy as np
import matplotlib.pyplot as plt

class NeuralNetworkMLP:
    def __init__(self):
        pass


class Layer():
    def __init__(self, input_shape, regularization, lr, n_neuron = 50, activation = 'sigmoid'):

        n_samples, n_features = input_shape

        self.lr = lr
        self.regularization = regularization
        self.n_neuron = 50
        self.input_shape = input_shape

        if activation == 'sigmoid':
            self.activation = self._sigmoid
            self.act_derivative = self._sigmoid_derivative

        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_neuron))
        self.bias = np.zeros(shape=(self.n_neuron))
        

    def _forward(self, X):
        
        self.z = np.dot(X, self.weights) + self.bias
        self.a = self._sigmoid(self.z)

        return self.a

    def _backward_inner(self, cost_derivative, delta, previous_a, weights_before):
        delta = np.dot(delta, weights_before.T) * self._sigmoid_derivative(self.z)

        sigma_bias = np.sum(delta, axis=0)
        sigma_weights = np.dot(previous_a.T, delta)

        self.weights -= self.lr * sigma_weights + self.regularization * self.weights
        self.bias -= self.lr * sigma_bias

        return delta

    def _backward_last(self, cost_derivative, previous_a, y):
        delta = cost_derivative(y, self.a) * self.act_derivative(self.z)

        sigma_bias = np.sum(delta, axis=0)
        sigma_weights = np.dot(previous_a.T, delta)

        self.weights -= self.lr * sigma_weights + self.regularization * self.weights
        self.bias -= self.lr * sigma_bias

        return delta

    def _sigmoid(self, z):
        return 1 / ( 1 + np.exp(-np.clip(z, -250, 250)))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1. - self._sigmoid(z))

if __name__ == '__main__':
    
    layer = Layer((3,5), regularization=0.05, lr=0.001)