#!/usr/bin/env python3
"""function that slices a matrix along specified axes"""

import numpy as np


def np_slice(matrix, axes=None):
    """function that slices a matrix along specified axes
    Args:
        matrix: the numpy.ndarray to slice
        axes: a dictionary of axes to slice along, where the key is the axis
              number and the value is a tuple of (start, stop, step)
              for slicing
    Returns:
        the sliced numpy.ndarray
    """
    if axes is None:
        axes = {}
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if not isinstance(axes, dict):
        raise TypeError("axes must be a dictionary")

    # Create a list of slice objects for each axis
    slices = [slice(None)] * matrix.ndim  # Start with full slices for all axes

    for axis, (start, stop, step) in axes.items():
        if not isinstance(axis, int) or axis < 0 or axis >= matrix.ndim:
            raise ValueError(f"Invalid axis: {axis}")
        if not (isinstance(start, int) or start is None):
            raise ValueError(f"Invalid start value for axis {axis}: {start}")
        if not (isinstance(stop, int) or stop is None):
            raise ValueError(f"Invalid stop value for axis {axis}: {stop}")
        if not (isinstance(step, int) or step is None):
            raise ValueError(f"Invalid step value for axis {axis}: {step}")

        slices[axis] = slice(start, stop, step)

    return matrix[tuple(slices)]
