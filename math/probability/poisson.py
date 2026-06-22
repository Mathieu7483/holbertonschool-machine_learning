#!/usr/bin/env python3
"""Create a class Poisson that represents a poisson distribution"""


class Poisson:
    """Represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initializes the Poisson distribution

        Args:
            data (list): list of the data to be used to estimate
            the distribution
            lambtha (float): expected number of occurences in a
            given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of
        “successes”

        Args: k (int): number of successes
        Returns: float: PMF value for k
        """
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        pmf = (e ** (-self.lambtha)) * (self.lambtha ** k) / self.factorial(k)
        return pmf

    def factorial(self, n):
        """Calculates the factorial of a number

        Args:
            n (int): number to calculate the factorial of
        Returns:
            int: factorial of n
        """
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n - 1)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of
        “successes”

        Args: k (int): number of successes
        Returns: float: CDF value for k
        """
        if k < 0:
            return 0
        k = int(k)
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
