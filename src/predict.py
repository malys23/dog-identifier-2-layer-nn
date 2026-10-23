#Predicts dog (1) or not dog (0) for each column of X with the 2 layer NN

import numpy as np
from forward_propagation import linear_activation_forward

def predict(X, parameters):
    m= X.shape[1]
    Y_prediction = np.zeros((1, m))
    A1, _ = linear_activation_forward(X, parameters["W1"], parameters["b1"], activation="relu")
    A2, _ = linear_activation_forward(A1, parameters["W2"], parameters["b2"], activation="sigmoid")
    
    for i in range(A2.shape[1]):
        if A2[0,i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction