#!/usr/bin/env python3
"""Write a function that computes the policy with a weight of a matrix."""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix.

    Args:
        matrix (numpy.ndarray): A numpy.ndarray of shape (m, n)
        containing the input matrix.
        weight (numpy.ndarray): A numpy.ndarray of shape (n, k)
        containing the weights.

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (m, k)
        containing the computed policy (probabilities).
    """
    logits = np.dot(matrix, weight)

    exp_scores = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
