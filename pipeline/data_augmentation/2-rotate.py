#!/usr/bin/env python3
"""write a function that rotates an image 90 degrees clockwise"""
import tensorflow as tf


def rotate_image(image):
    """rotates an image 90 degrees clockwise

    Args:
        image (tf.Tensor): image to rotate

    Returns:
        tf.Tensor: rotated image
    """
    return tf.image.rot90(image)
