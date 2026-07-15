#!/usr/bin/env python3
"""Write a function that performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number of
        data points and d is the number of dimensions in each point.
        ndim (int): Number of dimensions to project onto.

    Returns:
        numpy.ndarray: Weights matrix W of shape (d, ndim) where ndim is the
        new dimensionality of the transformed X.
    """
    # Compute the SVD of the data matrix
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Transposed (for shape(d, ndim)) top "ndim" rows of Vt
    return Vt[:ndim].T
