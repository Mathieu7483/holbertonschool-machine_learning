#!/usr/bin/env python3
"""write a function that performs a pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs a pooling on images

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
            c: number of channels in the images
        kernel_shape: tuple of (kh, kw) containing the kernel shape for
                      pooling
            kh: height of the kernel
            kw: width of the kernel
        stride: tuple of (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image
        mode: indicates the type of pooling
            max: indicates max pooling
            avg: indicates average pooling
    Returns: numpy.ndarray containing the pooled images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    pool_h = (h - kh) // sh + 1
    pool_w = (w - kw) // sw + 1
    pooled_images = np.zeros((m, pool_h, pool_w, c))
    for i in range(pool_h):
        for j in range(pool_w):
            img_slice = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            if mode == 'max':
                pooled_images[:, i, j] = np.max(img_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j] = np.mean(img_slice, axis=(1, 2))
    return pooled_images
