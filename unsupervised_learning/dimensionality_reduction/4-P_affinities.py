#!/usr/bin/env python3
"""Calculates the symmetric P affinities of a data set"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d)
        tol (float): Tolerance for binary search
        perplexity (float): Desired perplexity

    Returns:
        P (numpy.ndarray): Symmetric P affinities of shape (n, n)
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        betamin = None
        betamax = None
        Di = np.delete(D[i], i)
        Hi, Pi = HP(Di, betas[i])
        Hdiff = Hi - H
        tries = 0

        while abs(Hdiff) > tol and tries < 50:
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
            
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H
            tries += 1

        P[i, np.arange(n) != i] = Pi

    # Symmetrize P and normalize
    P = (P + P.T) / (2 * n)
    return P
