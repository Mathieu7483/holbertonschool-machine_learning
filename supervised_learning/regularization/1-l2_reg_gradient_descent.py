#!/usr/bin/env python3
"""Calculates the gradient of the cost with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Calculates the gradient of the cost with L2 regularization.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels.
        weights: dict of the weights and biases (numpy arrays).
        cache: dict of the outputs of each layer of the network.
        alpha: learning rate.
        lambtha: regularization parameter.
        L: number of layers in the network.

    Updates the weights and biases in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        dW = (np.matmul(dZ, cache['A' + str(i - 1)].T) / m
              + (lambtha * weights['W' + str(i)]) / m)
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dZ = (np.matmul(weights['W' + str(i)].T, dZ)
                  * (1 - cache['A' + str(i - 1)] ** 2))

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
