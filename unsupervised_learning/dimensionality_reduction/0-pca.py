#!/usr/bin/env python3
"""Write a function that performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
            n: number of data points
            d: number of dimensions in each point
        var: float, the fraction of variance that the PCA
         ptransformation should maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) containing the eigenvectors
            nd: new dimensionality of the transformed X
    """
    # Calculate the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the cumulative variance ratio
    cumulative_variance_ratio = (
        np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    )

    # Determine the number of components to retain
    # based on the desired variance
    num_components = np.searchsorted(cumulative_variance_ratio, var) + 1

    # Select the top 'num_components' eigenvectors
    W = sorted_eigenvectors[:, :num_components]

    return W
