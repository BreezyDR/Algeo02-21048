from typing import List, Optional
import numpy as np


###         JUST A LAYOUT        ###
# WE MIGHT USE NUMPY ARRAY INSTEAD #
class Matrix:
    # def __init__(self, sizeY, sizeX) -> None:
    #     if sizeX == None:
    #         sizeX = sizeY

    #     self.sizeX = sizeX
    #     self.sizeY = sizeY
    #     self.buffer = [[0 for i in range(self.sizeX)] for j in range(self.sizeY)]

    def __init__(self, matrix=None) -> None:
        if matrix != None:
            self.assign(matrix=matrix)
        else:
            self.assign(matrix=[[]])

    def assign(self, matrix: List[List[int]]):
        self.buffer = [[i for i in matrix[j]] for j in range(len(matrix))] #is a deepcopy

        self.adjustSize()

    def adjustSize(self):
        self.sizeY = len(self.buffer)
        
        if self.sizeY == 0:
            self.sizeX = 0
        else :
            self.sizeX = len(self.buffer[0])

    ##operational
    #everything is sbuject to change
    def addBy(self, matrix) :
        matrix1 = self.buffer
        matrix2 = matrix

        arr_result = []
        for i in range(len(matrix1)):
            arr_col = []
            for j in range(len(matrix1[0])):
                arr_col.append(matrix1[i][j] + matrix2[i][j])
            arr_result.append(arr_col)
        
        self.assign(arr_result)

    def subtractBy(self, matrix) :
        matrix1 = self.buffer
        matrix2 = matrix

        arr_result = []
        for i in range(len(matrix1)):
            arr_col = []
            for j in range(len(matrix1[0])):
                arr_col.append(matrix1[i][j] - matrix2[i][j])
            arr_result.append(arr_col)
        
        self.assign(arr_result)

    def multiplyBy(self, matrix) :
        matrix1 = self.buffer
        matrix2 = matrix

        arr_result = []
        for i in range(len(matrix1)):
            arr_col = []
            for j in range(len(matrix1[0])):
                arr_col.append(matrix1[i][j] * matrix2[i][j])
            arr_result.append(arr_col)
        
        
        self.assign(arr_result)

    def divideBy(self, matrix=None, scalar=None) :
        arr_result = []

        if matrix != None:
            matrix1 = self.buffer
            matrix2 = matrix

            
            for i in range(len(matrix1)):
                arr_col = []
                for j in range(len(matrix1[0])):
                    arr_col.append(matrix1[i][j] / matrix2[i][j])
                arr_result.append(arr_col)

        elif scalar != None and scalar != 0:
            matrix1 = self.buffer
            
            for i in range(len(matrix1)):
                arr_col = []
                for j in range(len(matrix1[0])):
                    arr_col.append(matrix1[i][j] / scalar)
                arr_result.append(arr_col)
        
        self.assign(arr_result)


    # linear algebra operational
    def transpose(self):
        trp = [[self.buffer[i][j] for i in range(self.sizeY)] for j in range(self.sizeX)]
        self.assign(trp)

    #deteminan with np
    def dummy_getdet(self) -> int :
        return np.linalg.det(np.array(self.buffer))

    def squashMat(m : List[List[int]]) -> List[int]:
        return [ i for j in range(len(m)) for i in m[j]]

    def describe(self):
        print('Matrix buffer: ')
        for i in range(self.sizeY):
            for j in range(self.sizeX):
                print(self.buffer[i][j], end='\t')
            print()
        print('sizeX :', self.getSizeX())
        print('sizeY :', self.getSizeY())
        print()


    #getter setter
    def getSizeX(self):
        return self.sizeX

    def getSizeY(self):
        return self.sizeY

    def getMatrix(self):
        return self.buffer



# static method
def identity_matrix(dimension : int) -> List[List[int]] :
    # Membuat matriks identitas sesuai dimensi matriks A
    identity = [[0 for j in range(dimension)] for i in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            if (i == j):
                identity[i][j] = 1
    return identity
