#!/usr/bin/env python3
"""Write a function that calculates the Shannon
entropy and P affinities relative to a data point"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point.

    Args:
        Di (numpy.ndarray):Pairwise squared Euclidean distances of shape(n-1,)
        beta (float): Precision value for the Gaussian kernel.

    Returns:
        H (float): Shannon entropy of the P affinities.
        Pi (numpy.ndarray): P affinities of shape (n-1,).
    """
    # Calculate numerator of probabilities
    num = np.exp(-Di * beta)
    sum_num = np.sum(num)

    # Avoid division by zero if sum_num is extremely small
    if sum_num == 0:
        Pi = np.zeros_like(Di)
    else:
        Pi = num / sum_num

    # Calculate Shannon entropy using base 2 logarithm, avoiding log2(0)
    H = -np.sum(Pi[Pi > 0] * np.log2(Pi[Pi > 0]))

    return H, Pi
