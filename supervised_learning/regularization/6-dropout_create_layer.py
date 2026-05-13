#!/usr/bin/env python3
"""Write a function that creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout
    Parameters:
        prev (tf.Tensor): Output from the previous layer
        n (int): Number of nodes in the layer to create
        activation: Activation function that the layer should use
        keep_prob (float): Probability that a node will be kept
        training (bool): Whether the model is training or not
    Returns:
        tf.Tensor: The output of the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    dense_layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer)

    dropout = tf.nn.dropout(dense_layer(prev), rate=1-keep_prob)

    return dropout
