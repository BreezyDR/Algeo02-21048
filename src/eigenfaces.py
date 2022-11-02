def mean(matrix):
    matriksnilaitengah = []
    for i in range(len(matrix)):
        sum = 0
        for j in range(len(matrix[0])):
            sum += matrix[i][j]
        matriksnilaitengah.append(sum / len(matrix[0]))
    return matriksnilaitengah


def subtract_matrix(matrix1, matrix2):
    arr_result = []
    for i in range(len(matrix1)):
        arr_col = []
        for j in range(len(matrix1[0])):
            arr_col.append(matrix1[i][j] - matrix2[i][j])
        arr_result.append(arr_col)
    return arr_result


def multiply_matrix(matrix1, matrix2):
    arr_result = [[0 for j in range(len(matrix2[0]))] for i in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(0, len(matrix2)):
                arr_result[i][j] = (matrix1[i][k] * matrix2[k][j])
    return arr_result


def euclidean_distance(vector1, vector2):
    sum = 0
    for i in range(len(vector1)):
        subtraction = vector1[i] - vector2[i]
        sum += subtraction**2
    return (sum**(1/2))