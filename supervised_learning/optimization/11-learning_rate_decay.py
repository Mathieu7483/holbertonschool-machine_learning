#!/usr/bin/env python3
"""Write the function def learning_rate_decay(alpha, decay_rate, global_step,
decay_step): that sets up learning rate decay for the training of a model:"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Sets up learning rate decay for the training of a model:

    Args:
        alpha: The original learning rate.
        decay_rate: The weight used to determine the rate at which alpha
            will decay.
        global_step: The number of passes of gradient descent that have
            elapsed.
        decay_step: The number of passes of gradient descent that should
            occur before alpha is decayed.
    Returns:
        The learning rate decayed by decay_rate every decay_step passes of
            gradient descent."""

    return alpha / (1 + decay_rate * (global_step // decay_step))
