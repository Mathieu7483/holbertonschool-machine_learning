#!/usr/bin/env python3
"""Write a class Neuron that defines a single neuron performing
binary classification (Based on 0-neuron.py):"""

import numpy as np


class Neuron:
    """Class Neuron that defines a single neuron
    performing binary classification"""

    def __init__(self, nx):
        """Constructor method for the class Neuron

        Args:
            nx: is the number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def A(self):
        """The getter method for the output of the neuron (prediction)"""
        return self.__A

    @property
    def W(self):
        """The getter method for the weights vector of the neuron"""
        return self.__W

    @property
    def b(self):
        """The getter method for the bias of the neuron"""
        return self.__b
