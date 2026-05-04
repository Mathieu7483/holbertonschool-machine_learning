#!/usr/bin/env python3
"""Write the function def shuffle_data(X, Y):
that shuffles the data points in two matrices the same way:"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way:
    Args:
        X: is the numpy.ndarray of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features
        Y: is the numpy.ndarray of shape (m, ny) to shuffle
            m is the number of data points
            ny is the number of features in Y
    Returns: The shuffled X and Y matrices, respectively
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
