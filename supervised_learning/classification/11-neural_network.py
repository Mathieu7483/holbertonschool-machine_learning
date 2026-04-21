#!/usr/bin/env python3
"""write a class NeuralNetwork that defines
a neural network with one hidden layer"""

import numpy as np


class NeuralNetwork:
    """Class NeuralNetwork that defines
     a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """Constructor method for the class NeuralNetwork

        Args:
            nx: is the number of input features to the neuron
            nodes: is the number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """The getter method for the weights vector of the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """The getter method for the bias of the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """The getter method for the output of the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """The getter method for the weights vector of the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """The getter method for the bias of the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """The getter method for the output of the output neuron (predict)"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X: is a numpy.ndarray with shape (nx, m) that contains the input
            data, where nx is the number of input features to the neuron and
            m is the number of examples

        Returns:
            The output of the neural network (prediction)
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data, where m is the number of examples
            A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example

        Returns:
            The cost of the model
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
