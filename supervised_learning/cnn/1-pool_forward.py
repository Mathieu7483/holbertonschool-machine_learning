#!/usr/bin/env python3
"""write a function that performs forward propagation
over a pooling layer of a neural network:"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network:
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
    pooling
        kh is the filter height
        kw is the filter width
    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height of the pooling
    sw is the stride for the width of the pooling
    mode is a string containing either max or avg, indicating whether to use
    maximum or average pooling, respectively
    Returns: a numpy.ndarray containing the output of the pooling layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_output = (h_prev - kh) // sh + 1
    w_output = (w_prev - kw) // sw + 1

    pooled = np.zeros((m, h_output, w_output, c_prev))

    for i in range(h_output):
        for j in range(w_output):
            region = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(region, axis=(1, 2))

    return pooled
