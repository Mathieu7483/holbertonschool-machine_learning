#!/usr/bin/env python3
"""function that calculates the derivative of a polynomial."""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial.
    poly is a list of coefficients representing the polynomial
    Returns a new list of coefficients representing the derivative
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    return [i * poly[i] for i in range(1, len(poly))]
