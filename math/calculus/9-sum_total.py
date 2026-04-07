#!/usr/bin/env python3
"""function that calculates the sum of i^2 for i from 1 to n."""


def summation_i_squared(n):
    """Calculates the sum of i^2 for i from 1 to n.
    n is the stopping condition"""
    if n is None or not isinstance(n, int):
        return None
    return n * (n + 1) * (2 * n + 1) // 6
