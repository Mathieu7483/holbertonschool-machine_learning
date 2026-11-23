#!/usr/bin/env python3
"""Write a function that crops an image to a given size"""
import tensorflow as tf


def crop_image(image, size):
    """Crops an image to a given size

    Args:
        image (tf.Tensor): image to crop
        size (tuple): size to crop to

    Returns:
        tf.Tensor: cropped image
    """
    return tf.image.resize_with_crop_or_pad(image, size[0], size[1])
