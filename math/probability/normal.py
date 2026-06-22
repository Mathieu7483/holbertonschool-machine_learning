#!/usr/bin/env python3
"""Create a class Normal that represents a normal distribution"""


class Normal:
    """Represents a normal distribution"""
    e = 2.7182818285  # Approximation of Euler's number
    pi = 3.1415926536  # Approximation of Pi

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes the Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum([(x - self.mean) ** 2 for x in data]) /
                           len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Parameters:
        x (float): The x-value.

        Returns:
        float: The PDF value for x.
        """
        coefficient = 1 / (self.stddev * (2 * Normal.pi) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return coefficient * (Normal.e ** exponent)
