#!/usr/bin/env python3
"""Write the function def update_variables_RMSProp"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using the RMSProp optimization algorithm.

    Parameters:
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    s is the previous second moment of var


    Returns:tuple containing the updated variable and the new squared gradient.
    """
    s = beta2 * s + (1 - beta2) * np.square(grad)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
