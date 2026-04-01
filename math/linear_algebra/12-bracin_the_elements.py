#!/usr/bin/env python3
"""file that contains the multiple element-wise operations"""


def np_elementwise(mat1, mat2):
    """function that performs element-wise addition, subtraction,
    multiplication, and division
    Returns:
        the element-wise addition, subtraction, multiplication,
        and division of the two matrices as a tuple
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
