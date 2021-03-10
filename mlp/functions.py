import numpy as np

"""
Activation functions with corresponding derivatives
"""

def sigmoid(z):
    return 1 / ( 1 + np.exp(-np.clip(z, -250, 250)))

def sigmoid_derivative(z):
    """
    Derivative of sigmoid function
    """
    return sigmoid(z) * (1. - sigmoid(z))

ACTIVATIONS = {
    'sigmoid' : sigmoid
}

ACT_DERIVATIVES = {
    'sigmoid' : sigmoid_derivative
}

"""
Loss functions with corresponding derivatives
"""

def sse(y, output):
    """
    Method computes a cost with sum of squared errors
    """
    cost = 1/2 * np.sum(np.square(output - y))
    return cost
    
def sse_derivative(y, output):
    """
    Derivative of quadratic cost
    """
    return (output - y)

def cross_entropy(y, output):
    """
    Method computes a cost with cross-entropy
    """
    cost = np.sum(-y * (np.log(output)) - (1. - y) * np.log(1. - output))
    return cost

def cross_entropy_derivative(y, output):
    """
    Derivative of cross-entropy cost
    """
    return (output - y) / ((1. - output) * output)

LOSS = {
    'sse' : sse,
    'cross-entropy' : cross_entropy
}

LOSS_DERIVATIVE = {
    'sse' : sse_derivative,
    'cross-entropy' : cross_entropy_derivative
}
