#!/usr/bin/env python3
"""
Functions to save and load model weights.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights.

    Args:
        network: The model whose weights should be saved.
        filename: The path of the file to save the weights to.
        save_format: The format in which the weights should be saved.

    Returns:
        None
    """
    try:
        network.save_weights(filename, save_format=save_format)
    except TypeError:
        network.save_weights(filename)

    return None


def load_weights(network, filename):
    """
    Loads a model's weights.

    Args:
        network: The model to which the weights should be loaded.
        filename: The path of the file to load the weights from.

    Returns:
        None
    """
    network.load_weights(filename)
    return None