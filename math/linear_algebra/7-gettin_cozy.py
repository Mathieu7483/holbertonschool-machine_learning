#!/usr/bin/env python3
"""function that concatenates two matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices

    Args:
        mat1: first matrix
        mat2: second matrix
        axis: axis to concatenate along

    Returns:
        new matrix
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_matrix = []
        for i in range(len(mat1)):
            new_matrix.append(mat1[i] + mat2[i])
        return new_matrix
