#!/usr/bin/env python3
"""Write a function def grads(Y, P): that calculates the gradients of Y"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): Dataset of shape (n, ndim)
        containing the low dimensional transformation of X.
        P (numpy.ndarray): Array of shape (n, n)
        containing the P affinities of X.

    Returns:
        dY (numpy.ndarray): Array of shape (n, ndim)
        containing the gradients of Y.
        Q (numpy.ndarray): Array of shape (n, n)
        containing the Q affinities of Y.
    """
    # Compute the Q affinities using the Q_affinities function
    Q, num = Q_affinities(Y)

    # Compute the difference between P and Q
    PQ_diff = P - Q

    # Compute the gradients dY
    dY = np.dot(PQ_diff, Y)

    return dY, Q
