#!/usr/bin/env python3
"""function that calculates the sum of i^2 for i from 1 to n."""


def summation_i_squared(n):
    """Calculates the sum of i^2 for i from 1 to n.
    n is the stopping condition"""
    if n is None or not isinstance(n, int):
        return None
    if n < 1:
        return 0
    return sum(i**2 for i in range(1, n + 1))
