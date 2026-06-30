#!/usr/bin/env python3
"""Write a function that calculates the minor matrix of a matrix"""


def minor(matrix):
    """Calculates the minor matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
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

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            sub_mat = [row[:j] + row[j+1:] for k,
                       row in enumerate(matrix) if k != i]
            minor_row.append(determinant(sub_mat))
        minor_matrix.append(minor_row)

    return minor_matrix
