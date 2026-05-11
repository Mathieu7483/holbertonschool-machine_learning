#!/usr/bin/env python3
"""Module for calcultaing the F1 score from a confusion matrix."""
import numpy as np
precision = __import__('2-precision').precision
specificity = __import__('3-specificity').specificity


def f1_score(confusion):
    """
    Calculates the F1 score from a confusion matrix.
    Args:
    confusion (numpy.ndarray): A confusion matrix of shape (classes, classes)
                                    with row indices representing the correct
                                    labels and column indices representing the
                                    predicted labels.
    Returns:
        numpy.ndarray: A numpy.ndarray of shape (classes,) containing the
                        F1 score of each class.
    """
    # Calculate precision and specificity for each class
    prec = precision(confusion)
    spec = specificity(confusion)

    # Calculate recall for each class using specificity
    recall = spec

    # Calculate F1 score for each class
    f1 = 2 * (prec * recall) / (prec + recall)

    return f1
