#!/usr/bin/env python3
"""write a function that performs forward propagation
over a convolutional layer of a neural network:"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1, 1)):
    """performs forward propagation over a convolutional
     layer of a neural network:
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid,
    indicating the type of padding used
    Returns: a numpy.ndarray containing the output
    of the convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
    elif padding == 'valid':
        pad_h = 0
        pad_w = 0

    h_output = (h_prev + 2 * pad_h - kh) // sh + 1
    w_output = (w_prev + 2 * pad_w - kw) // sw + 1

    A_padded = np.pad(A_prev,
                      ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                      mode='constant')

    convolved = np.zeros((m, h_output, w_output, c_new))

    for i in range(h_output):
        for j in range(w_output):
            region = A_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            for k in range(c_new):
                convolved[:, i, j, k] = np.sum((region * W[:, :, :, k]),
                                               axis=(1, 2, 3))

    return activation(convolved + b)
