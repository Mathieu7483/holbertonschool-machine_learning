#!/usr/bin/env python3
"""Write the function def create_Adam_op(alpha, beta1, beta2, epsilon):
that sets up the Adam optimization algorithm in TensorFlow:"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Sets up the Adam optimization algorithm in TensorFlow:

    Args:
        alpha: The learning rate.
        beta1: The weight used for the first moment.
        beta2: The weight used for the second moment.
        epsilon: A small number to avoid division by zero.

    Returns:
        An Adam optimization operation.
    """
    return tf.keras.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1,
                                    beta_2=beta2,
                                    epsilon=epsilon)
