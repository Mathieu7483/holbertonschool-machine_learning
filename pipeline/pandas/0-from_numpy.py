#!/usr/bin/env python3
"""Write a function that creates a pd.DataFrame from a np.ndarray"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray"""
    columns = [chr(i) for i in range(65, 65 + array.shape[1])]
    df = pd.DataFrame(array, columns=columns)
    return df
