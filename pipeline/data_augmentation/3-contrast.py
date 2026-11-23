#!/usr/bin/env python3
"""Write a function that contrasts an image by a given factor"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Contrasts an image by a given factor

    Args:
        image (tf.Tensor): image to contrast
        lower (float): lower bound of contrast factor
        upper (float): upper bound of contrast factor

    Returns:
        tf.Tensor: contrasted image
    """
    return tf.image.random_contrast(image, lower, upper)
