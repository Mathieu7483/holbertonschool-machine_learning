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
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixels values are between 0 and 1
           and its largest side is 512 pixels"""
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
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

        resized_image = tf.image.resize(
            image, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC
        )
        clipped_image = tf.clip_by_value(resized_image, 0.0, 255.0)
        scaled_image = clipped_image / 255.0
        return scaled_image[tf.newaxis, :]

    def load_model(self):
        """Creates the model used to calculate cost.
        The model uses the VGG19 Keras model as a base. The input of the model
        is the same as the VGG19 input, and the output is a list containing
        the outputs of the VGG19 layers listed in style_layers followed by
        content_layer.

        Returns:
        model (tf.keras.Model): The Keras model used to calculate cost."""
        # Load VGG19 model from Keras API
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        vgg.trainable = False
        # Replace MaxPooling2D layers with AveragePooling2D layers
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        # get outputs of the style and content layers from modified VGG19
        style_outputs = [vgg.get_layer(
            name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        # Create the model, make it non-trainable and return it
        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=style_outputs + [content_output])
