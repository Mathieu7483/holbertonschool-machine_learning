#!/usr/bin/env python3
"""Write a function that hues an image by a given factor"""
import tensorflow as tf


def change_hue(image, delta):
    """Hues an image by a given factor

    Args:
        image (tf.Tensor): image to hue
        factor (float): factor to hue by

    Returns:
        tf.Tensor: hued image
    """
    return tf.image.adjust_hue(image, delta)
