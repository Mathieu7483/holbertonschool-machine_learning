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
        # Extract distances for point i, excluding the point itself (diagonal)
        idx = np.concatenate((np.r_[0:i], np.r_[i + 1:n]))
        Di = D[i, idx]

        # Initialize boundaries for binary search
        betamin = None
        betamax = None
        
        # First calculation of entropy and conditional affinities
        Hi, Pi = HP(Di, betas[i])
        Hdiff = Hi - H
        tries = 0

        # Binary search to find the optimal beta_i for the target perplexity
        while abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                # Entropy is too high -> beta needs to increase (adjust lower bound)
                betamin = betas[i]
                if betamax is None:
                    betas[i] *= 2.0
                else:
                    betas[i] = (betas[i] + betamax) / 2.0
            else:
                # Entropy is too low -> beta needs to decrease (adjust upper bound)
                betamax = betas[i]
                if betamin is None:
                    betas[i] /= 2.0
                else:
                    betas[i] = (betas[i] + betamin) / 2.0

            # Recalculate with the new beta_i value
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H
            tries += 1

        # Insert calculated conditional affinities, leaving the diagonal at 0
        P[i, idx] = Pi

    # Symmetrize and normalize the P affinity matrix
    P = (P + P.T) / (2 * n)

    return P
