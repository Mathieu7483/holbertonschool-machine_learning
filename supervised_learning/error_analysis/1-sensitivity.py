#!/usr/bin/env python3
"""Module for calculating sensitivity from a confusion matrix."""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity from a confusion matrix.
    Args:
    confusion (numpy.ndarray): A confusion matrix of shape (classes, classes)
                                    with row indices representing the correct
                                    labels and column indices representing the
                                    predicted labels.
    Returns:
        numpy.ndarray: A numpy.ndarray of shape (classes,) containing the
                        sensitivity of each class.
    """
    # True Positives are the diagonal elements of the confusion matrix
    TP = np.diag(confusion)

    # False Negatives are the sum of each row minus the True Positives
    FN = np.sum(confusion, axis=1) - TP

    # Calculate sensitivity for each class
    sensitivity = TP / (TP + FN)

    return sensitivity
