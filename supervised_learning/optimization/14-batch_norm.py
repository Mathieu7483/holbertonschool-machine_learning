#!/usr/bin/env python3
"""Write the function def create_batch_norm_layer(prev, n, activation):
that creates a batch normalization layer for
a neural network in tensorflow:"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow.

    Args:
        prev: The output from the previous layer.
        n: The number of nodes in the layer to be created.
        activation: The activation function that should be used on the
        output of the layer.

    Returns:
        The activated output of the batch normalization layer."""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)
    batch_norm = tf.keras.layers.BatchNormalization()(dense)
    return activation(batch_norm)
