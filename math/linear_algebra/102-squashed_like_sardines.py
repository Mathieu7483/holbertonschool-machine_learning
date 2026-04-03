#!/usr/bin/env python3
"""
function that concatenates two matrices along a specific axis without using
"""
import numpy as np

def cat_matrices(mat1, mat2, axis=0):
    """
    Concatène deux matrices avec NumPy.
    """
    try:
        # np.concatenate prend un tuple de matrices en premier argument
        return np.concatenate((mat1, mat2), axis=axis)
    except (ValueError, TypeError):
        # NumPy lève une ValueError si les dimensions ne correspondent pas
        return None
