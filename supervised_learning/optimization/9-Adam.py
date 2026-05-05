#!/usr/bin/env python3
"""Write the function def update_variables_Adam
(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
that updates a variable in place using the Adam
optimization algorithm:"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm:

    Args:
        alpha: The learning rate.
        beta1: The weight used for the first moment.
        beta2: The weight used for the second moment.
        epsilon: A small number to avoid division by zero.
        var: A numpy.ndarray containing the variable to be updated.
        grad: A numpy.ndarray containing the gradient of var.
        v: A numpy.ndarray containing the previous first moment of var.
        s: A numpy.ndarray containing the previous second moment of var.
        t: The time step used for bias correction.

    Returns:
        The updated variable, the new first moment, and the new second
            moment, respectively.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * np.square(grad)
    v_corrected = v / (1 - np.power(beta1, t))
    s_corrected = s / (1 - np.power(beta2, t))
    var -= alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
