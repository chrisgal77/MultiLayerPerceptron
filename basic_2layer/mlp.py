import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt 
"""
input X - [n_samples, n_features]

y - [n_samples]

weights = [n_hidden, n_features]

"""

class NeuralNetworkMLP:
    """
    Simple neural network with 2 layers.
    """
    def __init__(self, lr = 0.01, epochs = 100, l2 = 0, batch_size=4):
        self.lr = lr
        self.epochs = epochs
        self.n_hidden = 100
        self.l2 = l2
        self.batch_size = batch_size

    def fit(self, X, y, valid = None):
        
        n_output = np.unique(y).shape[0]
        n_samples, n_features = X.shape

        self.weights_hidden = np.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        self.bias_hidden = np.zeros(shape=(self.n_hidden))

        self.weights_out = np.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))
        self.bias_out = np.zeros(shape=(n_output))

        y = self._onehot(n_output, y)

        _costs = []
        if valid:
            X_valid, y_valid = valid
            _valid_acc = []
            _train_acc = []

        for _ in tqdm(range(self.epochs)):

            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for index in range(0, indices.shape[0] - self.batch_size + 1, self.batch_size):

                batches_indices = indices[index:index + self.batch_size]

                z_hidden, a_hidden, z_out, a_out = self._forward(X[batches_indices])

                #backpropagation

                # delta = self._cost_SSE_derivative(y[batches_indices], a_out) * self._sigmoid_derivative(z_out)
                delta = self._cost_CE_derivative(y[batches_indices], a_out) * self._sigmoid_derivative(z_out)

                sigma_bias_out = np.sum(delta, axis=0)
                sigma_weights_out = np.dot(a_hidden.T, delta)

                delta = np.dot(delta, self.weights_out.T) * self._sigmoid_derivative(z_hidden)

                sigma_bias_hidden = np.sum(delta, axis=0)
                sigma_weights_hidden = np.dot(X[batches_indices].T, delta)

                self.weights_out -= self.lr * sigma_weights_out + self.l2 * self.weights_out
                self.bias_out -= self.lr * sigma_bias_out

                self.weights_hidden -= self.lr * sigma_weights_hidden + self.l2 * self.weights_hidden
                self.bias_hidden -= self.lr * sigma_bias_hidden

            z_hidden, a_hidden, z_out, a_out = self._forward(X)

            # cost = self._cost_SSE(y, a_out)
            cost = self._cost_CE(y, a_out)

            _costs.append(cost)
            try:
                y_valid_pred = self.predict(X_valid)
                _valid_acc.append((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])
            except:
                pass
        
        plt.plot(range(self.epochs), _costs)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.tight_layout()
        plt.show()

        plt.plot(range(self.epochs), _valid_acc, color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.show()

        return self
        

    def _sigmoid(self, z):
        return 1 / ( 1 + np.exp(-np.clip(z, -250, 250)))


    def _sigmoid_derivative(self, z):
        """
        Derivative of sigmoid function
        """
        return self._sigmoid(z) * (1. - self._sigmoid(z))


    def _forward(self, X):
        
        z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden

        a_hidden = self._sigmoid(z_hidden)

        z_out = np.dot(a_hidden, self.weights_out) + self.bias_out

        a_out = self._sigmoid(z_out)

        return z_hidden, a_hidden, z_out, a_out


    def _onehot(self, n_output, y):
        """
        Method encodes labels to one-hot representation
        """
        onehot = np.zeros((n_output, y.shape[0]))
        for i, value in enumerate(y):
            onehot[value, i] = 1.
        
        return onehot.T


    def _cost_SSE(self, y, output):
        """
        Method computes a cost with sum of squared errors
        """

        l2 = self.l2 * (np.sum((self.weights_hidden)**2) + np.sum((self.weights_out)**2))
        cost = 1/2 * np.sum(np.square(output - y)) + l2
        return cost


    def _cost_SSE_derivative(self, y, output):
        """
        Derivative of quadratic cost
        """
        return (output - y)

    
    def _cost_CE(self, y, output):
        """
        Method computes a cost with cross-entropy
        """
        l2 = self.l2 * (np.sum((self.weights_hidden)**2) + np.sum((self.weights_out)**2))
        term1 = -y * (np.log(output))
        term2 = (1. - y) * np.log(1. - output)
        cost = np.sum(term1 - term2) + l2
        return cost


    def _cost_CE_derivative(self, y, output):
        """
        Derivative of cross-entropy cost
        """
        return (output - y) / ((1. - output) * output)


    def predict(self, X):
        z_hidden, a_hidden, z_output, a_output = self._forward(X)

        return np.argmax(z_output, axis=1)

        
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def bad_ones(y_true, y_pred):
    pass

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nn = NeuralNetworkMLP(epochs=100, batch_size=2)

nn.fit(X_train, y_train, (X_test, y_test))

pred = nn.predict(X_test)
print(accuracy(y_test, pred))
