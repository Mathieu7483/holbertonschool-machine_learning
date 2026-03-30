#!/usr/bin/env python3
""" function that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise
    Args:
        arr1: first array to add
        arr2: second array to add
    Returns:
        a new array containing the element-wise sum of arr1 and arr2
    """
    if len(arr1) != len(arr2):
        return None

    return [arr1[i] + arr2[i] for i in range(len(arr1))]
