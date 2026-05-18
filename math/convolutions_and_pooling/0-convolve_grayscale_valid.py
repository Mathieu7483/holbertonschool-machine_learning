#!/usr/bin/env python3
"""write a function that performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images

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
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_h = h - kh + 1
    conv_w = w - kw + 1
    convolved_images = np.zeros((m, conv_h, conv_w))
    for i in range(conv_h):
        for j in range(conv_w):
            convolved_images[:, i, j] = np.sum(
                images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))
    return convolved_images
