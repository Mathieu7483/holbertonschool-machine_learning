#!/usr/bin/env python3
"""Write a function that initializes all variables
required to calculate the P affinities in t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number of
        data points and d is the number of dimensions in each point.
        perplexity (float): Desired perplexity for the P affinities.

    Returns:
        D (numpy.ndarray): Pairwise squared Euclidean distances of shape (n, n)
        P (numpy.ndarray): P affinities of shape (n, n).
        betas (numpy.ndarray): Precision values of shape (n, 1).
        H (float): Shannon entropy of the P affinities.
    """
    n, d = X.shape
    # Compute the pairwise squared Euclidean distances
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)  # Set diagonal to zero to avoid self-distances

    # Initialize variables
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)  # Initial Shannon entropy based on perplexity

    return D, P, betas, H
