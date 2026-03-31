#!/usr/bin/env python3
"""function that multiplies two matrices"""


def mat_mul(mat1, mat2):
    """multiplies two matrices

    Args:
        mat1: first matrix
        mat2: second matrix

    Returns:
        new matrix
    """
    if len(mat1[0]) != len(mat2):
        return None
    new_matrix = []
    for i in range(len(mat1)):
        new_matrix.append([])
        for j in range(len(mat2[0])):
            new_matrix[i].append(0)
            for k in range(len(mat1[0])):
                new_matrix[i][j] += mat1[i][k] * mat2[k][j]
    return new_matrix
