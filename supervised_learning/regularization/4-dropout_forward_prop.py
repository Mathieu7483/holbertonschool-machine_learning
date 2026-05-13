#!/usr/bin/env python3
"""Write a function that conducts forward propagation using Dropout:"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout:
    X is the input data for the network
    weights is a dictionary of the weights and biases of the neural network
    L is the number of layers in the network
    keep_prob is the probability that a node will be kept
    Returns: a dictionary containing the outputs of each layer and the dropout
        masks used on each layer, respectively"""
    cache = {}
    A = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.matmul(W, A) + b

        if i == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)

            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob

            cache['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache
