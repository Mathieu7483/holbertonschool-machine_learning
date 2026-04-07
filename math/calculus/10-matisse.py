#!/usr/bin/env python3
"""function that calculates the derivative of a polynomial."""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial.
    poly is a list of coefficients representing the polynomial
    Returns a new list of coefficients representing the derivative
    """
    if not isinstance(poly, list):
        return None
    if not all(isinstance(coef, int) for coef in poly):
        return None
    if len(poly) == 1:
        return [0]
    return [coef * i for i, coef in enumerate(poly)][1:]
