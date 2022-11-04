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

    def __init__(self, matrix) -> None:
        self.assign(matrix=matrix)


    

    def assign(self, matrix: List[List[int]]):
        self.buffer = [[i for i in matrix[j]] for j in range(len(matrix))] #is a deepcopy

        self.adjustSize()

    def adjustSize(self):
        self.sizeY = len(self.buffer)
        
        if self.sizeY == 0:
            self.sizeX = 0
        else :
            self.sizeX = len(self.buffer[0])


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


    #getter setter
    def getSizeX(self):
        return self.sizeX

    def getSizeY(self):
        return self.sizeY

    def getMatrix(self):
        return self.buffer