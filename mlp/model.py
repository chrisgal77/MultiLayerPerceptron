import numpy as np
from tqdm import tqdm
from preprocessing.encoders import OneHotEncoder
from functions import LOSS, LOSS_DERIVATIVE
import matplotlib.pyplot as plt 
from metrics.classification import accuracy_score, cross_val_score
from layer import Layer

class MultiLayerPerceptron:
    def __init__(self, lr = 0.001, l2 = 0.01, epochs = 50, batch_size = 16, loss = 'sse'):
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size

        self.loss = LOSS[loss]
        self.loss_derivative = LOSS_DERIVATIVE[loss]

        self.layers = []

    def fit(self, X, y):
        
        y_enc = OneHotEncoder().encode(y)
        _cost = []
        for _ in tqdm(range(self.epochs)):

            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)         

            for index in range(0, indices.shape[0] - self.batch_size + 1, self.batch_size):

                batches_indices = indices[index:index + self.batch_size]

                self._forward(X[batches_indices])
                self._backprop(X[batches_indices], y_enc[batches_indices])

            self._forward(X)
            _cost.append(self.loss(y_enc, self.a[-1]))
        
        # plt.plot(range(self.epochs), _cost)
        # plt.show()
        
        return self

    def add_layer(self, n_neuron = 30, activation = 'sigmoid', input_length = None):
        if not self.layers and not input_length:
            raise ValueError('First layer must have an input length')
        if not input_length: 
            self.layers.append(Layer(n_neuron, self.layers[-1].n_neuron, lr = self.lr, activation = activation))
        else:
            self.layers.append(Layer(n_neuron, input_length, lr = self.lr, activation = activation))

    def _forward(self, X):

        self.z, self.a = [], []
        z,a = self.layers[0].forward(X)
        self.z.append(z)
        self.a.append(a)
        for idx in range(1, len(self.layers)):
            z, a = self.layers[idx].forward(a)
            self.z.append(z)
            self.a.append(a)
   
    def _backprop(self, X, y):
        
        delta = self.loss_derivative(y, self.a[-1]) * self.layers[-1].act_derivative(self.z[-1])
        for idx in reversed(range(1, len(self.layers))):
            delta = self.layers[idx].backward(delta, self.a[idx - 1], self.z[idx - 1])
        self.layers[0].backward(delta, X)

    def predict(self, X):
        self._forward(X)
        return np.argmax(self.z[-1] , axis=1)
  
def build_model():
    
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    X, y = load_digits(return_X_y=True)
    n_output = np.unique(y).shape[0]
    n_samples, n_features = X.shape
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    nn = MultiLayerPerceptron(loss='cross-entropy', epochs=100, batch_size=4)
    nn.add_layer(50, 'relu', input_length=n_features)
    nn.add_layer(n_output, 'sigmoid')
    
    nn.fit(X_train, y_train)
    pred = nn.predict(X_test)
    print(accuracy_score(y_test, pred))
    
    # print(cross_val_score(nn, X, y))  
  
if __name__ == '__main__':
    
    build_model()