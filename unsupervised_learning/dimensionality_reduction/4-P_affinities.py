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
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        betas[i] = 1.0
        betamin = None
        betamax = None
        Di = np.delete(D[i], i)
        Hdiff = 1
        tries = 0

        while abs(Hdiff) > tol and tries < 50:
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H
            if Hdiff > 0:
                betamin = betas[i]
                if betamax is None:
                    betas[i] *= 2.0
                else:
                    betas[i] = (betas[i] + betamax) / 2.0
            else:
                betamax = betas[i]
                if betamin is None:
                    betas[i] /= 2.0
                else:
                    betas[i] = (betas[i] + betamin) / 2.0
            tries += 1

        P[i, np.arange(n) != i] = Pi

    # Symmetrize P and normalize
    P = (P + P.T) / (2 * n)
    return P
