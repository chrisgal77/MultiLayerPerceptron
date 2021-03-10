import numpy as np 

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)