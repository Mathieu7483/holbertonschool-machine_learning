#!/usr/bin/env python3
"""matrix transpose module, must return a new matrix, never empty,
and assume all the elements in the same dimension are of the same type/shape"""


def matrix_transpose(matrix):
    """calculates the transpose of a matrix
    Args:
        matrix: list of lists whose transpose should be calculated
    Returns:
        new matrix representing the transpose of matrix
    """
    if not matrix or not matrix[0]:
        return []

    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
