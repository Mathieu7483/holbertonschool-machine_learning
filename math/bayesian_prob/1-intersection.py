#!/usr/bin/env python3
"""Based on likelihood.py write a function
that calculates the intersection of obtaining
this data with the various hypothetical
probabilities:"""
import numpy as np


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining
    this data with the various hypothetical
    probabilities:
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
      probabilities of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns: a 1D numpy.ndarray containing the intersection of
      obtaining x and n with each probability in P, respectively"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    binom_coeff = (np.math.factorial(n) /
                   (np.math.factorial(x) * np.math.factorial(n - x)))
    likelihood = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    intersection = likelihood * Pr

    return intersection
