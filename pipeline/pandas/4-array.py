#!/usr/bin/env python3
"""Write a function def array(df): that takes
a pd.DataFrame as input and performs the following"""


def array(df):
    """The function should select the last 10

    rows of the High and Close column
    and return them as a 2D numpy.ndarray
    """
    return df[["High", "Close"]].tail(10).to_numpy()
