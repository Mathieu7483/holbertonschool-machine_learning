#!/usr/bin/env python3
"""Module for calculating precision from a confusion matrix."""
import numpy as np


def precision(confusion):
    """
    Calculates the precision from a confusion matrix.
    Args:
    confusion (numpy.ndarray): A confusion matrix of shape (classes, classes)
                                    with row indices representing the correct
                                    labels and column indices representing the
                                    predicted labels.
    Returns:
        numpy.ndarray: A numpy.ndarray of shape (classes,) containing the
                        precision of each class.
    """
    # True Positives are the diagonal elements of the confusion matrix
    TP = np.diag(confusion)

    # False Positives are the sum of each column minus the True Positives
    FP = np.sum(confusion, axis=0) - TP

    # Calculate precision for each class
    precision = TP / (TP + FP)

    return precision
