#!/usr/bin/env python3
"""
function that concatenates two matrices along a specific axis without using
"""


def get_shape(matrix):
    """Calculate the shape of the matrix ."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape


def cat_matrices(mat1, mat2, axis=0):
    """
    concatenate two matrices along a specific axis without using external libraries.
    """
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # 1. check if the number of dimensions is the same
    if len(shape1) != len(shape2):
        return None

    # 2. check if the dimensions match for all axes
    #  except the one we're concatenating on
    for i in range(len(shape1)):
        if i != axis:
            if shape1[i] != shape2[i]:
                return None

    # 3. recursive function to concatenate matrices at the specified axis
    def recurse_cat(m1, m2, current_axis):
        """recursive function to concatenate matrices at the specified axis"""
        if current_axis == 0:
            # base case: concatenate along the current axis
            # return m1 + m2 if both are lists, otherwise return None
            return m1 + m2

        # otherwise, we need to go deeper into the structure
        new_matrix = []
        for i in range(len(m1)):
            res = recurse_cat(m1[i], m2[i], current_axis - 1)
            new_matrix.append(res)
        return new_matrix

    try:
        return recurse_cat(mat1, mat2, axis)
    except (TypeError, IndexError):
        # Secure against cases where m1 or m2 are
        # not lists or have different structures
        return None