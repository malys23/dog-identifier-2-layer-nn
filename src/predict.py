#Predicts dog (1) or not dog (0) for each column of X with the 2 layer NN

import numpy as np
from forward_propagation import linear_activation_forward

def predict(X, parameters):
    A1, _ = linear_activation_forward(X, parameters["W1"], parameters["b1"], activation="relu")
    A2, _ = linear_activation_forward(X, parameters["W2"], parameters["b2"], activation="sigmoid")
    return (A2 > 0.5).astype(int)