#!/user/bin/env python3
"""Create a class Poisson that represents a poisson distribution"""


class Poisson:
    """Represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initializes the Poisson distribution

        Args:
            data (list): list of the data to be used to estimate the distribution
            lambtha (float): expected number of occurences in a given time frame
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
