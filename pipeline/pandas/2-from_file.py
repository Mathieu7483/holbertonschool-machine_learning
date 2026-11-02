#!/usr/bin/env python3
"""Create a pd.DataFrame from a file"""
import pandas as pd


def from_file(filename, delimiter):
    """Creates a pd.DataFrame from a file"""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
