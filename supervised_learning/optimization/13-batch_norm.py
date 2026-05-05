#!/usr/bin/env python3
"""Write the function def batch_norm(Z, gamma, beta, epsilon):
that normalizes an unactivated output of a
neural network using batch normalization:"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch
    normalization.

    Args:
        Z: A numpy.ndarray of shape (m, n) containing the unactivated output
            of a neural network.
        gamma: A numpy.ndarray of shape (1, n) containing the scales used for
            batch normalization.
        beta: A numpy.ndarray of shape (1, n) containing the offsets used for
            batch normalization.
        epsilon: A small number to avoid division by zero.

    Returns:
        The normalized Z matrix."""
    m, n = Z.shape
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
