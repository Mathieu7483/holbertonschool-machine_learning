#!/usr/bin/env python3
"""function that slices a matrix along specified axes"""


def np_slice(matrix, axes=None):
    """
    Args:
        matrix: numpy.ndarray of any shape
        axes: dict of the form {axis: (start, stop, step)}
    Returns:        sliced numpy.ndarray along the specified axes
    """
    import numpy as np

    
    list_slice = [slice(None)] * matrix.ndim

    for axis , values in axes.items():
        list_slice[axis] = slice(*values)
    return matrix[tuple(list_slice)]
