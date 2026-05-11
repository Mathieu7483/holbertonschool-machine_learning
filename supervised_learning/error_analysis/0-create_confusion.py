#!/usr/bin/env python3
"""0-create_confusion.py"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    m, n = logits.shape
    confusion = np.zeros((n, n))
    for i in range(m):
        j = np.argmax(logits[i])
        confusion[labels[i], j] += 1
    return confusion
