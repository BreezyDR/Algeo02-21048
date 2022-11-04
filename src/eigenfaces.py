import numpy as np
import math

from typing import Optional, List

def mean(matrix):
    # matrix = [A_1, A_2, ... A_len(matrix)]
    # dengan A: Vektor hasil linierisasi
    # Menghitung elemen rata-rata dari tiap elemen i dari A
    # dan memasukkannya ke matriksnilaitengah
    matriksnilaitengah = []
    for i in range(len(matrix[0])):
        sum = 0
        for j in range(len(matrix)):
            sum += matrix[i][j]
        matriksnilaitengah.append(sum / len(matrix[0]))
    return matriksnilaitengah

# def multiply_matrix(matrix1, matrix2):
#     return np.matmul(matrix1, matrix2)


def euclidean_distance(vector1 : List[int], vector2 : List[int]) -> List[int] :
    euclidean_array = []
    for i in range(len(vector1)):
        sum = 0
        subtraction = vector1[i] - vector2[i]
        sum += subtraction**2
        sum = math.sqrt(sum)
        euclidean_array.append(sum)
    return euclidean_array


def min_value(array):
    # Mengembalikan indeks di mana nilai terkecil
    return min(array)
    

# def transpose_matrix(matrix):
#     # Membuat transpos suatu matriks
#     return np.transpose(matrix)


# def identity_matrix(dimension : int) -> List[List[int]] :
#     # Membuat matriks identitas sesuai dimensi matriks A
#     identity = [[0 for j in range(dimension)] for i in range(dimension)]
#     for i in range(dimension):
#         for j in range(dimension):
#             if (i == j):
#                 identity[i][j] = 1
#     return identity

# def determinant(matrix : List[List[int]]) -> int :
#     # Mencari determinan suatu matriks
#     # Prekondisi: Matriks harus persegi
#     return np.linalg.det(matrix)
