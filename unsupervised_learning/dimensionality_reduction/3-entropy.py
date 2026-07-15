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
    # Compute the P affinities using the Gaussian kernel
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)

    # Normalize the P affinities
    Pi /= sum_Pi

    # Calculate the Shannon entropy
    H = -np.sum(Pi * np.log(Pi))

    return H, Pi
