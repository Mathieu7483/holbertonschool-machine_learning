#!/usr/bin/env python3
"""Write the function def normalize(X, m, s)0:
that normalizes (standardizes) a matrix:"""
import numpy as np


def normalize(X, m, s):
    """normalizes (standardizes) a matrix:
    Args:
        X: is the numpy.ndarray of shape (m, n) to normalize
            m is the number of data points
            n is the number of features
        m: is a numpy.ndarray of shape (n,) containing the mean of each feature
        s: is a numpy.ndarray of shape (n,) containing the standard
           deviation of each feature
    Returns: The normalized X matrix
    """
    return (X - m) / s
