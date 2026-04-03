#!/usr/bin/env python3
"""function that concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specific axis
    Returns:
        the concatenated matrix
    """
    if axis == 0:
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return None
        return mat1 + mat2
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        sub_cat = cat_matrices(mat1[i], mat2[i], axis - 1)

        if sub_cat is None:
            return None
        result.append(sub_cat)

    return result
