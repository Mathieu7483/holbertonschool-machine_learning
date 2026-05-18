#!/usr/bin/env python3
"""write a function that performs a padding convolution on grayscale images"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a padding convolution on grayscale images

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
                the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding: tuple of (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image
    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    conv_h = h + 2 * ph - kh + 1
    conv_w = w + 2 * pw - kw + 1
    convolved_images = np.zeros((m, conv_h, conv_w))
    for i in range(conv_h):
        for j in range(conv_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))
    return convolved_images
