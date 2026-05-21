#!/usr/bin/env python3
"""Write a function that performs back propagation
over a pooling layer of a neural network:"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network:
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing
    the size of the kernel for the pooling
        kh is the filter height
        kw is the filter width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height of the pooling
        sw is the stride for the width of the pooling
    mode is a string that indicates the type of pooling
    that will be performed, either max or avg
    Returns: dA_prev"""
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    m, h_new, w_new, c_new = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            if mode == 'max':
                for k in range(c):
                    A_slice = A_prev[:, vert_start:vert_end,
                                     horiz_start:horiz_end, k]
                    mask = (A_slice == np.max(A_slice, axis=(1, 2),
                                              keepdims=True))
                    dA_prev[:, vert_start:vert_end, horiz_start:horiz_end,
                            k] += (mask * dA[:, i, j, k][:, np.newaxis,
                                                         np.newaxis])
            elif mode == 'avg':
                # Backward Average Pooling
                da = dA[:, i, j, :]
                shape = (m, kh, kw, c)
                avg_val = da[:, np.newaxis, np.newaxis, :] / (kh * kw)
                dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, :] += (
                    np.tile(avg_val, (1, kh, kw, 1)))

    return dA_prev
