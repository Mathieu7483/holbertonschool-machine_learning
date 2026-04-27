#!/usr/bin/env python3
"""converts a numeric label vector into a one-hot matrix"""

from tensorflow import keras as K

def one_hot(labels, classes=None):
  """converts a numeric label vector into a one-hot matrix"""
  return K.utils.to_categorical(labels, num_classes=classes)
