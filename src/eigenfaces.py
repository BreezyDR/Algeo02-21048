from typing import List, Optional
import numpy as np

# def mean(matrix: List[List[int]]):
#     # matrix = [A_1, A_2, ... A_len(matrix)]
#     # dengan A: Vektor hasil linierisasi
#     # Menghitung elemen rata-rata dari tiap elemen i dari A
#     # dan memasukkannya ke matriksnilaitengah
#     matriksnilaitengah = []
#     for i in range(len(matrix[0])):
#         sum = 0
#         for j in range(len(matrix)):
#             sum += matrix[i][j]
#         matriksnilaitengah.append(sum / len(matrix[0]))
#     return matriksnilaitengah


# def subtract_matrix(matrix1 : List[List[int]], matrix2 : List[List[int]]) -> List[int] :
#     arr_result = []
#     for i in range(len(matrix1)):
#         arr_col = []
#         for j in range(len(matrix1[0])):
#             arr_col.append(matrix1[i][j] - matrix2[i][j])
#         arr_result.append(arr_col)
#     return arr_result


# def multiply_matrix(matrix1 : List[List[int]], matrix2 : List[List[int]]) -> List[int] :
#     arr_result = [[0 for j in range(len(matrix2[0]))] for i in range(len(matrix1))]
#     for i in range(len(matrix1)):
#         for j in range(len(matrix2[0])):
#             for k in range(0, len(matrix2)):
#                 arr_result[i][j] = (matrix1[i][k] * matrix2[k][j])
#     return arr_result


def euclidean_distance(vector1 : List[int], vector2 : List[int]) -> List[int] :
    euclidean_array = []
    for i in range(len(vector1)):
        sum = 0
        subtraction = vector1[i] - vector2[i]
        sum += subtraction**2
        sum = sum**(1/2)
        euclidean_array.append(sum)
    return euclidean_array


def search_value(array : List[int], value : int) -> int :
    # Mengembalikan indeks terkecil sebuah value yang dicari.
    # Jika tidak ada, mengembalikan index -1.
    index = -1
    for i in range(len(array)):
        if (array[i] == value):
            index = i
            break
    return index


def min_value(array : List[int]) -> int :
    # Mengembalikan indeks di mana nilai terkecil
    min = array[0]
    for i in range(len(array)):
        if (min < array[i]):
            array[i] = min
    index = search_value(array, min)
    return index
    

# def transpose_matrix(matrix : List[List[int]]) -> List[List[int]] :
#     # Membuat transpos suatu matriks
#     arr_result = [[0 for j in range(len(matrix))] for i in range(len(matrix[0]))]
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             arr_result[i][j] = matrix[j][i]
#     return arr_result


# def identity_matrix(dimension : int) -> List[List[int]] :
#     # Membuat matriks identitas sesuai dimensi matriks A
#     identity = [[0 for j in range(dimension)] for i in range(dimension)]
#     for i in range(dimension):
#         for j in range(dimension):
#             if (i == j):
#                 identity[i][j] = 1
#     return identity

# ini lom bisa
def determinant(matrix : List[List[int]]) -> int :
    # Mencari determinan suatu matriks
    # Prekondisi: Matriks harus persegi
    temp = [[0 for j in range(len(matrix[0]))] for i in range(len(matrix))]
    tanda = 1
    det = 0
    if (len(matrix) == 1):
        det = matrix[0][0]
    else:
        for i in range(len(matrix)):
            for j in range(1, len(matrix)):
                for k in range(len(matrix[0])):
                    if (k < i):
                        temp[j-1][k] = matrix[j][k]
                    elif (k > i):
                        temp[j-1][k-1] = matrix[j][k]
            det += tanda*matrix[0][i] * determinant(temp)
            tanda = -tanda
    return det