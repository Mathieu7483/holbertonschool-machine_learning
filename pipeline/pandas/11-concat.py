#!/usr/bin/env python3
"""Write a function def concat(df1, df2): that takes two pd.DataFrame objects:

Indexes both dataframes on their Timestamp columns.
Includes all timestamps from df2 (bitstamp) up to and including timestamp
1417411920.
Concatenates the selected rows from df2 to the top of df1 (coinbase).
Adds keys to the concatenated data, labeling the rows from df2 as bitstamp
and the rows from df1 as coinbase.
You should use index = __import__('10-index').index
Returns the concatenated pd.DataFrame."""


def concat(df1, df2):
  """Concatenates two pd.DataFrame objects"""
  index = __import__("10-index").index
  df1 = index(df1)
  df2 = index(df2)
  df2 = df2[df2.index <= 1417411920]
  return __import__("pandas").concat([df2, df1], keys=["bitstamp", "coinbase"])
