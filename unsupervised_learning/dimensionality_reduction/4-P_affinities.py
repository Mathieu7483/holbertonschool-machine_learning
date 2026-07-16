#!/usr/bin/env python3
"""Write a function that calculates the symmetric P affinities of a data set"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number of
        data points and d is the number of dimensions in each point.
        tol (float): Tolerance for the binary search to achieve the desired
        perplexity.
        perplexity (float): Desired perplexity for the P affinities.

    Returns:
        P (numpy.ndarray): Symmetric P affinities of shape (n, n).
    """
    n, d = X.shape
    # Initialize variables
    D, P, betas, _ = P_init(X, perplexity)

    # Calculate P affinities for each data point
    for i in range(n):
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        H, Pi = HP(Di, betas[i])
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = Pi

    # Symmetrize the P affinities
    P = (P + P.T) / (2 * n)

    return P
