#!/usr/bin/env python3
"""write a function that performs forward propagation
over a convolutional layer of a neural network:"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same'):
    """performs forward propagation over a convolutional layer of a neural network:
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
    Returns: a numpy.ndarray containing the output of the convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape

    if padding == 'same':
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
    else:
        pad_h = 0
        pad_w = 0

    A_padded = np.pad(A_prev,
                      ((0,), (pad_h,), (pad_w,), (0,)),
                      mode='constant')

    h_new = h_prev + 2 * pad_h - kh + 1
    w_new = w_prev + 2 * pad_w - kw + 1

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h
                    vert_end = h + kh
                    horiz_start = w
                    horiz_end = w + kw

                    A_slice = A_padded[i,
                                       vert_start:vert_end,
                                       horiz_start:horiz_end,
                                       :]
                    Z[i, h, w, c] = np.sum(A_slice * W[:, :, :, c
                    ]) + b[:, :, :, c]
    A = activation(Z)
    return A
