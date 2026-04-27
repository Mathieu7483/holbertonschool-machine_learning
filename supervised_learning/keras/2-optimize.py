#!/usr/bin/env python3
"""
Sets up Adam optimization for a Keras model with categorical crossentropy loss.
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with
    categorical crossentropy loss.

    Args:
        network (keras.Model): The Keras model to optimize.
        alpha (float): The learning rate.
        beta1 (float): The beta1 parameter for the Adam optimizer.
        beta2 (float): The beta2 parameter for the Adam optimizer.
    """
    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                  beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
