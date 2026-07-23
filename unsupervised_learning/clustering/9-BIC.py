#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a Gaussian Mixture Model using the
    Bayesian Information Criterion (BIC).
    """
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

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax < kmin or kmax > n:
        return None, None, None, None

    if kmin > kmax:
        return None, None, None, None

    b = []
    likelihoods = []
    best_bic = float('inf')
    best_k = None
    best_results = None

    for k in range(kmin, kmax + 1):
        pi, m, S, g, li = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None or m is None or S is None or g is None or li is None:
            return None, None, None, None

        # p: number of parameters
        p = (k * d) + (k * d * (d + 1) // 2) + (k - 1)
        bic = p * np.log(n) - 2 * li

        likelihoods.append(li)
        b.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_results = (pi, m, S)
            best_k = k

    return best_k, best_results, np.array(likelihoods), np.array(b)
