import numpy as np

class OneHotEncoder:
    def __init__(self):
        pass

    def encode(self, y):
        """
        Method encodes labels to one-hot representation.
        """
        n_output = np.unique(y).shape[0]
        onehot = np.zeros((n_output, y.shape[0]))

        for i, value in enumerate(y):
            onehot[value, i] = 1.
        
        return onehot.T


        
