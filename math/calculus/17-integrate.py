#!/usr/bin/env python3
"""functoion that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial

    Args:
        poly: list of coefficients representing the polynomial
        C: integration constant (default is 0)

    Returns:
        list of coefficients representing the integral of the polynomial
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]
    for i in range(len(poly)):
        integral.append(poly[i] / (i + 1))
    return integral
