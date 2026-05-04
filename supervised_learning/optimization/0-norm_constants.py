#!/usr/bin/env python3
"""Write the function def normalization_constants(X):
that calculates the normalization (standardization)
constants of a matrix:"""
import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization)
    constants of a matrix:
    Args:
        X: is the numpy.ndarray of shape (m, n) to normalize
            m is the number of data points
            n is the number of features
    Returns: The mean and standard deviation of each feature, respectively,
             as a tuple of numpy.ndarrays of shape (n,)
    """
    m, n = X.shape
    mean = np.mean(X, axis=0)
    standard = np.std(X, axis=0)
    return mean, standard
