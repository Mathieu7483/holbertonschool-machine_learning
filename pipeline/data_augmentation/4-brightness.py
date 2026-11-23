#!/usr/bin/env python3
"""Write a function that brightens an image by a given factor"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Brightens an image by a given factor

    Args:
        image (tf.Tensor): image to brighten
        max_delta (float): maximum delta to brighten by

    Returns:
        tf.Tensor: brightened image
    """
    return tf.image.random_brightness(image, max_delta)
