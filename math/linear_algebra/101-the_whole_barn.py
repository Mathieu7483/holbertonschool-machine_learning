#!/usr/bin/env python3
"""function that adds two matrices"""


def add_matrices(mat1, mat2):
    """function that adds two matrices
    Args:
        mat1: first matrix
        mat2: second matrix
    Returns:
        the sum of the two matrices
        or None if the two matrices cannot be added
    """
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            sum_values = add_matrices(mat1[i], mat2[i])
            if sum_values is None:
                return None
            result.append(sum_values)
        return result
    
    try:
        return mat1 + mat2
    except TypeError:
        return None
     