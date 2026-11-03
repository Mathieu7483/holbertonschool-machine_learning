#!/usr/bin/env python3
"""Computes descriptive statistics of a DataFrame"""


def analyze(df):
    """Computes descriptive statistics for all columns except Timestamp"""
    return df.drop(columns=['Timestamp']).describe()
