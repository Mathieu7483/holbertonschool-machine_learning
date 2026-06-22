#!/usr/bin/env python3
"""Create a class Binomail that represents a binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the Binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.n = len(data)
            self.p = sum(data) / (self.n * max(data))
