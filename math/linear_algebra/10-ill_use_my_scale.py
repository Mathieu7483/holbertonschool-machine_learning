#!/usr/bin/env python3
"""file that contains the function np_shape"""


import numpy as np


def np_shape(matrix):
    """function that calculates the shape of a matrix

    Args:
        matrix: is a numpy.ndarray of any shape

    Returns:
        the shape of the matrix as a tuple
    """
    return matrix.shape
