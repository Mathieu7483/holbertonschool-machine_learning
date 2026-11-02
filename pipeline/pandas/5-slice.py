#!/usr/bin/env python3
"""Write a function def slice(df): that takes a pd.DataFrame and:

Extracts the columns High, Low, Close, and Volume_(BTC).
Selects every 60th row from these columns.
Returns: the sliced pd.DataFrame"""


def slice(df):
    """Slices a pd.DataFrame"""
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
