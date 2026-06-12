#!/usr/bin/env python3
"""Create a class NST that performs tasks for neural style transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Class NST that performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Constructor method
        Args:
            style_image:preprocesed style image
            content_image: preprocessed content image
            alpha: the weight for the content cost
            beta: the weight for the style cost
        """
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    def scale_image(self, image):
        """Rescales an image such that its pixels values are between 0 and 1
           and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or \
           image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))
        resized_image = tf.image.resize(image, (new_h, new_w),
                                        method=tf.image.ResizeMethod.AREA)
        scaled_image = resized_image / 255.0
        return scaled_image[tf.newaxis, :]
