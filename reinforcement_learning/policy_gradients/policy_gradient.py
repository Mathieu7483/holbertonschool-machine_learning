#!/usr/bin/env python3
"""Write a function that computes the policy with a weight of a matrix."""


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix.

    Args:
        matrix (numpy.ndarray): A numpy.ndarray of shape (m, n) containing
            the input data.
        weight (numpy.ndarray): A numpy.ndarray of shape (n, k) containing
            the weights.

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (m, k)
        containing the computed policy.
    """
    return matrix @ weight
