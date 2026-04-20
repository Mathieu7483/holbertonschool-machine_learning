#!/usr/bin/env python3
"""Write a class Neuron that defines a single neuron performing
binary classification (Based on 1-neuron.py):"""

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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X: is a numpy.ndarray with shape (nx, m) that contains the input
            data, where nx is the number of input features to the neuron and
            m is the number of examples

        Returns:
            The output of the neuron (prediction)
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data, where m is the number of examples
            A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
