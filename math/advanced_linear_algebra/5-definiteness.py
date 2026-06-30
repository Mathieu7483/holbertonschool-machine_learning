#!/usr/bin/env python3
"""Write a function that calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.size == 0:
        return None

    # 2. Control of symmetry
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    # 3. Treatement of eigenvalues with a tolerance to avoid floating-point
    atol = 1e-7

    # Extraction of eigenvalues based on the tolerance
    pos = eigenvalues > atol
    neg = eigenvalues < -atol
    zero = np.isclose(eigenvalues, 0, atol=atol)

    # 4. Classification of the matrix based on the eigenvalues
    if np.all(pos):
        return "Positive definite"
    elif np.all(pos | zero) and np.any(zero):
        return "Positive semi-definite"
    elif np.all(neg):
        return "Negative definite"
    elif np.all(neg | zero) and np.any(zero):
        return "Negative semi-definite"
    elif np.any(pos) and np.any(neg):
        return "Indefinite"

    return None
