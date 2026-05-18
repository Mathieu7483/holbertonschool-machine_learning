#!/usr/bin/env python3
"""write a function that performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same'):
    """performs a convolution on images with channels

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
            c: number of channels in the images
        kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel for
                the convolution
            kh: height of the kernel
            kw: width of the kernel
            c: number of channels in the kernel
        padding: is either a tuple of (ph, pw), ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
            if a tuple:
                ph: padding for the height of the image
                pw: padding for the width of the image
    Returns: numpy.ndarray containing the convolved images"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    if c != kc:
        raise ValueError("The number of channels in the kernel must match "
                         "the number of channels in the images")
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    conv_h = h + 2 * ph - kh + 1
    conv_w = w + 2 * pw - kw + 1
    convolved_images = np.zeros((m, conv_h, conv_w))
    for i in range(conv_h):
        for j in range(conv_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2, 3))
    return convolved_images
