#!/usr/bin/env python3
"""function that slices a matrix along specified axes"""

import numpy as np


def slice_matrix(matrix, axes=None):
    """
    Args:
        matrix: numpy.ndarray of any shape
        axes: dict of the form {axis: (start, stop, step)}
    Returns:        sliced numpy.ndarray along the specified axes
    """
    if axes is None:
        axes = {}
    
    r_start, r_stop, r_step = axes.get(0, (None, None, None))
    c_start, c_stop, c_step = axes.get(1, (None, None, None))

    sub_matrix = matrix[slice(r_start, r_stop, r_step)]
    
    result = [ligne[slice(c_start, c_stop, c_step)] for ligne in sub_matrix]
    
    return result
