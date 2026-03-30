#!/usr/bin/env python3
"""Size me, please! a module that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """calculates the shape of a matrix
    Args:
        matrix: list of lists whose shape should be calculated
    Returns:
        tuple of integers representing the shape of the matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        return []

    return [len(matrix)] + matrix_shape(matrix[0])
