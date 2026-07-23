#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a Gaussian Mixture Model using the
    Bayesian Information Criterion (BIC).

    Parameters:
    - X (numpy.ndarray): The dataset to be analyzed, of shape (n, d).
    - kmin (int, optional): Minimum number of clusters to check. Default is 1.
    - kmax (int, optional): Maximum number of clusters to check. If None,
      it defaults to n (total number of data points).
    - iterations (int, optional): Maximum EM iterations. Default is 1000.
    - tol (float, optional): Tolerance for convergence. Default is 1e-5.
    - verbose (bool, optional): If True, prints log likelihood info.

    Returns:
    - best_k (int): Optimal number of clusters based on BIC.
    - best_result (tuple): (pi, m, S) for the best cluster configuration.
    - likelihoods (numpy.ndarray): Log likelihoods for each k tested.
    - b (numpy.ndarray): BIC values for each k tested.

    Returns (None, None, None, None) on failure.
    """
    # 1. Type and value validation for standard arguments
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    # 2. Validation: kmin cannot be greater than total data points n
    if kmin > n:
        return None, None, None, None

    # 3. Handle default kmax value
    if kmax is None:
        kmax = n

    # 4. Validate kmax type and boundaries
    if not isinstance(kmax, int) or kmax < kmin or kmax > n:
        return None, None, None, None

    b = []
    likelihoods = []
    best_bic = float('inf')
    best_k = None
    best_results = None

    # 5. Iterate through each cluster size from kmin to kmax
    for k in range(kmin, kmax + 1):
        # Perform EM algorithm for current k
        pi, m, S, g, li = expectation_maximization(
            X, k, iterations, tol, verbose)

        # Check for algorithm failure or invalid outputs
        if pi is None or m is None or S is None or g is None or li is None:
            return None, None, None, None

        # Number of parameters p for a full covariance GMM:
        # - (k - 1) for independent priors
        # - (k * d) for cluster means
        # - (k * d * (d + 1) / 2) for full symmetric covariance matrices
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

        # Compute Bayesian Information Criterion: p * ln(n) - 2 * L
        bic = p * np.log(n) - 2 * li

        likelihoods.append(li)
        b.append(bic)

        # Update best configuration if current BIC is lower (lower is better)
        if bic < best_bic:
            best_bic = bic
            best_results = (pi, m, S)
            best_k = k

    return best_k, best_results, np.array(likelihoods), np.array(b)
