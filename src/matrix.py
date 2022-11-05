from typing import List, Optional
import numpy as np


###         JUST A LAYOUT        ###
# WE MIGHT USE NUMPY ARRAY INSTEAD #
class Matrix:
    def __init__(self, matrix : List[List[int]] = None) -> None:
        if matrix != None:
            self.assign(matrix=matrix)
        else:
            self.assign(matrix=[[]])

    def assign(self, matrix: List[List[int]]) -> None :
        self.buffer = [[i for i in matrix[j]] for j in range(len(matrix))] #is a deepcopy

        self.adjustSize()

    def adjustSize(self) -> None :
        self.sizeY = len(self.buffer)
        
        if self.sizeY == 0:
            self.sizeX = 0
        else :
            self.sizeX = len(self.buffer[0])

    ##operational
    #everything is sbuject to change
    def addBy(self, matrix : List[List[int]] = None) -> None :
        if matrix != None:
            arr_result = []
            for i in range(len(self.buffer)):
                arr_col = []
                for j in range(len(self.buffer[0])):
                    arr_col.append(self.buffer[i][j] + matrix[i][j])
                arr_result.append(arr_col)
            
            self.assign(arr_result)

    def subtractBy(self, matrix : List[List[int]] = None) -> None :
        if matrix != None:
            arr_result = []
            for i in range(len(self.buffer)):
                arr_col = []
                for j in range(len(self.buffer[0])):
                    arr_col.append(self.buffer[i][j] - matrix[i][j])
                arr_result.append(arr_col)
            
            self.assign(arr_result)

    def multiplyBy(self, matrix : Optional[List[List[int]]] = None, scalar : Optional[int] = None) -> None :
        arr_result = []

        if matrix != None:
            self.assign(np.matmul(self.buffer, matrix))
            return

        elif scalar != None and scalar != 0:
            matrix1 = self.buffer
            
            for i in range(len(matrix1)):
                arr_col = []
                for j in range(len(matrix1[0])):
                    arr_col.append(matrix1[i][j] / scalar)
                arr_result.append(arr_col)
        
        self.assign(arr_result)

    def divideBy(self, scalar : int = None) -> None :
        arr_result = []
        
        if scalar != None and scalar != 0:
            matrix1 = self.buffer
            
            for i in range(len(matrix1)):
                arr_col = []
                for j in range(len(matrix1[0])):
                    arr_col.append(matrix1[i][j] / scalar)
                arr_result.append(arr_col)
        
        self.assign(arr_result)


    # linear algebra operational
    def transpose(self) -> None :
        # trp = [[self.buffer[i][j] for i in range(self.sizeY)] for j in range(self.sizeX)]
        # self.assign(trp)
        self.assign(np.transpose(self.buffer))

    #deteminan with np
    def getDeterminant(self) -> int :
        return np.linalg.det(np.array(self.buffer))

    def getSquashedMatrix(m : List[List[int]]) -> List[int]:
        return [ i for j in range(len(m)) for i in m[j]]


    #tulis
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
def generateIdentityMatrix(dimension : int) -> List[List[int]] :
    # Membuat matriks identitas sesuai dimensi matriks A
    identity = [[0 for j in range(dimension)] for i in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            if (i == j):
                identity[i][j] = 1
    return identity