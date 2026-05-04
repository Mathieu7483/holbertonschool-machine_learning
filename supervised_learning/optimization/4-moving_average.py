#!/usr/bin/env python3
"""Write the function def moving_average(data, beta):
that calculates the weighted moving average of a data set:"""
import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    if not isinstance(data, list) or len(data) == 0:
        return None
    if not isinstance(beta, float) or beta < 0 or beta > 1:
        return None

    m_avg = []
    v = 0

    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        m_avg.append(v / (1 - beta ** (i + 1)))

    return m_avg
