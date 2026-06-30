#!/usr/bin/env python3
"""Write a function that calculates the inverse matrix of a matrix"""


def inverse(matrix):
    """Calculates the inverse matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    def determinant(sub_mat):
        """Calculates the determinant of a matrix"""
        if len(sub_mat) == 1:
            return sub_mat[0][0]
        if len(sub_mat) == 2:
            return (sub_mat[0][0] * sub_mat[1][1] -
                    sub_mat[0][1] * sub_mat[1][0])

        det = 0
        for i in range(len(sub_mat)):
            minor_sub_matrix = [row[:i] + row[i+1:] for row in sub_mat[1:]]
            det += ((-1) ** i) * sub_mat[0][i] * determinant(minor_sub_matrix)
        return det

    det = determinant(matrix)
    if det == 0:
        return None

    if len(matrix) == 1:
        return [[1 / matrix[0][0]]]

    n = len(matrix)
    inverse_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            minor_mat = [[matrix[x][y] for y in range(n) if y != j]
                         for x in range(n) if x != i]

            cofactor = ((-1) ** (i + j)) * determinant(minor_mat)
            inverse_matrix[j][i] = cofactor / det

    return inverse_matrix
