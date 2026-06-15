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
        self.generate_features()

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
        """
        # Load the VGG19 model
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        # Extract the outputs of the specified style and content layers
        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        # Build the model that outputs the style and content features
        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=style_outputs + [content_output]
        )

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of an input layer.

        Args:
            input_layer (tf.Tensor or tf.Variable): A tensor of shape
            (1, h, w, c) containing the layer output whose gram matrix
            should be calculated.

        Raises:
            TypeError: If input_layer is not a tensor of rank 4.

        Returns:
            tf.Tensor: A tensor of shape (1, c, c) containing the gram matrix
            of input_layer.
        """
        # Calidate input_layer rank and batch size
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        # Calculate the gram matrix using einsum for efficient computation
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Normalization by the number of locations (h * w)
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return gram / nb_locations

    def generate_features(self):
        """
        Extract the features used to calculate neural style cost.
        Sets the public instance attributes:
            - gram_style_features - a list of gram matrices calculated from the
                style layer outputs of the style image
            - content_feature - the content layer output of the content image
        """
        # Ensure the images are preprocessed correctly
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0)

        # Inférence unique par image
        style_outputs_raw = self.model(preprocessed_style)
        content_outputs_raw = self.model(preprocessed_content)

        # Slicing propre sur les listes de tenseurs obtenues
        style_outputs = style_outputs_raw[:-1]
        self.content_feature = content_outputs_raw[-1]

        # Calcul des matrices de Gram
        self.gram_style_features = [self.gram_matrix(output)
                                    for output in style_outputs]
