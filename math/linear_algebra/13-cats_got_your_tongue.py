#!/usr/bin/env python3
"""file that contains the function np_cat
that concatenates two matrices along a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specific axis
    Returns:
        the concatenated matrix
    """
    return np.concatenate((mat1, mat2), axis=axis)
