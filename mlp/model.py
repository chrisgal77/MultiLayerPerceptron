import numpy as np

from preprocessing.encoders import OneHotEncoder
from functions import LOSS, LOSS_DERIVATIVE
from layer import Layer


class MultiLayerPerceptron:
    def __init__(self, lr = 0.001, l2 = 0.01, epochs = 50, batch_size = 4, loss = 'sse'):
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size

        self.loss = LOSS[loss]
        self.loss_derivative = LOSS_DERIVATIVE[loss]

        self.layers = []

    def fit(self, X, y):

        n_output = np.unique(y).shape[0]
        n_samples, n_features = X.shape

        y = OneHotEncoder().encode(y)          


    def add_layer(self, n_neuron = 30, activation = 'sigmoid', input_length = None):
        if not self.layers and not input_length:
            raise ValueError('First layer must have an input length')
        if not input_length: 
            self.layers.append(Layer(n_neuron, self.layers[-1].n_neuron, activation))
        else:
            self.layers.append(Layer(n_neuron, input_length, activation))

    def forward(self, X):
        pass

    def backprop(self):
        pass

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.layers[-1].a, axis=1)

if __name__ == '__main__':

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    print(X.shape)

    nn = MultiLayerPerceptron()
    nn.add_layer(30, 'sigmoid')

    print(nn.layers)