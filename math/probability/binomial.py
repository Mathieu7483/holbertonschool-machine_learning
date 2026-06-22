#!/usr/bin/env python3
"""Create a class Binomail that represents a binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the Binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est

            self.n = n_est
            self.p = p_est

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes k"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        from math import comb
        return comb(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))
