#!/usr/bin/env python3
"""Write the function def update_variables_momentum
(alpha, beta1, var, grad, v): that updates a variable using
 the gradient descent with momentum optimization algorithm:"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Update a variable using the gradient descent with momentum
    optimization algorithm.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight.
    var (numpy.ndarray): The variable to be updated.
    grad (numpy.ndarray): The gradient of the variable.
    v (numpy.ndarray): The previous velocity.

    Returns:
    tuple: A tuple containing the updated variable and the new velocity.
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
