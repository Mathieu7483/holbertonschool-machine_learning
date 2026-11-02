#!/usr/bin/env python3
"""Write a function that takes a pd.DataFrame
as input and performs the following"""
import pandas as pd


def rename(df):
    """Renames the columns of a pd.DataFrame"""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
