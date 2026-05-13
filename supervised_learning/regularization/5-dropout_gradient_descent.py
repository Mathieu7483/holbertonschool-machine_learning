#!/usr/bin/env python3
"""Write a function that updates the weights of a neural network
with Dropout regularization using gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent
    Parameters:
        Y (numpy.ndarray): One-hot numpy.ndarray of shape (classes, m) that
            contains the correct labels for the data.
        weights (dict): Dictionary of the weights and biases.
        cache (dict): Dictionary of the outputs and dropout masks from
            forward propagation.
        alpha (float): Learning rate.
        keep_prob (float): Probability that a node will be kept.
        L (int): Number of layers in the network.
    Returns:
        None: Updates weights in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            D = cache['D' + str(layer - 1)]
            dA_prev = np.dot(W.T, dZ)
            dA_prev = np.multiply(dA_prev, D) / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
