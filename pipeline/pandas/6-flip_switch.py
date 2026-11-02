#!/usr/bin/env python3
"""Write a function def flip_switch(df): that takes a pd.DataFrame and:

Sorts the data in reverse chronological order.
Transposes the sorted dataframe.
Returns: the transformed pd.DataFrame."""


def flip_switch(df):
    """Flips a pd.DataFrame"""
    return df.sort_index(ascending=False).transpose()
