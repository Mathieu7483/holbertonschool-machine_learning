#!/usr/bin/env python3
"""function that adds two matrices"""


def add_matrices(mat1, mat2):
    """function that adds two matrices
    Args:
        mat1: first matrix
        mat2: second matrix
    Returns:
        the sum of the two matrices
    """
    if not mat1 or not mat2:
        return None

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    result = [[mat1[i][j] + mat2[i][j]
               for j in range(len(mat1[0]))] for i in range(len(mat1))]

    return result
