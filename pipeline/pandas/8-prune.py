#!/usr/bin/env python3
"""Write a function def prune(df): that takes a pd.DataFrame and:

Removes any entries where Close has NaN values.
Returns: the modified pd.DataFrame."""


def prune(df):
    """Removes any entries where Close has NaN values"""
    return df.dropna(subset=["Close"])
