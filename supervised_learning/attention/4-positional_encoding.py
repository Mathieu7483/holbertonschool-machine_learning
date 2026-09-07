#!/usr/bin/env python3
"""Write the function def positional_encoding(max_seq_len, dm):
that calculates the positional encoding for a transformer:
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Function that calculates the positional encoding for a transformer."""
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads
