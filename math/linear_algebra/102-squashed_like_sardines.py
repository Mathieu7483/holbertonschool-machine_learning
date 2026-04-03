#!/usr/bin/env python3
"""function that concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specific axis
    Returns:
        the concatenated matrix
    """
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
        return result
    else:
        return None
