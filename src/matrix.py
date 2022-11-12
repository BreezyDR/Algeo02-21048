from typing import List, Optional
from numpy.typing import ArrayLike

import numpy as np
import sympy as sp


class Matrix:
    def __init__(self, matrix : ArrayLike = None) -> None:
        if matrix != None:
            self.assign(matrix=matrix)

    def assign(self, matrix: ArrayLike) -> None :
        self.buffer = np.array(matrix)

        self.adjustSize()

    def adjustSize(self) -> None :
        self.sizeX = self.buffer.shape[0]
        self.sizeY = self.buffer.shape[1]

    def addBy(self, matrix : ArrayLike = None) -> None :
        if matrix is None:
            return

        matrix = np.array(matrix)

        if self.buffer.shape != matrix.shape :
            print('IN Matrix.py, ADDED 2 ARRAYS OF DIFFERENT SHAPE')
        
        self.assign(np.add(self.buffer, matrix))

    def subtractBy(self, matrix : ArrayLike = None) -> None :
        if matrix is None:
            return

        matrix = np.array(matrix)

        if self.buffer.shape != matrix.shape :
            print('IN Matrix.py, SUBTRACTED 2 ARRAYS OF DIFFERENT SHAPE')

        self.assign(np.subtract(self.buffer, matrix))

    def multiplyBy(self, matrix : Optional[ArrayLike] = None, scalar : Optional[int] = None) -> None :
        if matrix is None and scalar == None :
            return

        matrix = np.array(matrix)

        if self.buffer.shape != matrix.shape and matrix.shape != (1, 1) and scalar == None:
            print('IN Matrix.py, MULTIPLIED 2 ARRAYS OF DIFFERENT SHAPE')

        if matrix is not None:
            self.assign(np.multiply(self.buffer, matrix))
        elif scalar != None:
            self.assign(np.multiply(self.buffer, scalar))


    def divideBy(self, scalar : int = None) -> None :
        if scalar != 0 and scalar != None:
            self.assign(np.divide(self.buffer, scalar))
        
    # linear algebra operational
    def transpose(self) -> None :
        self.assign(np.transpose(self.buffer))

    #deteminan with np
    def getDeterminant(self) -> int :
        return np.linalg.det(np.array(self.buffer))

    #tulis
    def describe(self) -> None :
        print('Matrix buffer: ')
        print(self.buffer)
        print('sizeX :', self.getSizeX())
        print('sizeY :', self.getSizeY())
        print()

    #getter setter
    def getSizeX(self) -> int :
        return self.sizeX

    def getSizeY(self) -> int :
        return self.sizeY

    def getMatrix(self) -> np.ndarray :
        return self.buffer

    def isSquare(self) -> bool:
        return (self.buffer.shape[0] == self.buffer.shape[1])


    # static methods
    def getSquashedMatrix(m : ArrayLike) -> np.ndarray:
        return np.array(m).reshape(-1, 1)

    def getEigenValues(self, real = True) -> List[int]:
        """ 
        deprecated, too slow
        =======

        Menghasilkan list berupa eigen values yang sudah terurut secara descending.
        Akar-akar imajiner akan diabaikan."""
        if not self.isSquare():
            raise Exception("Matrix must be a square matrix")
        
        A = sp.Matrix(self.buffer)
        lamda = sp.Symbol("lamda", real=True)
        larr = sp.eye(len(self.buffer))*lamda
        B = larr-A

        eigenValues = sp.solve(B.det(),lamda)
        if real:
            eigenValues = list(map(float,eigenValues))
        eigenValues = np.flip(np.sort(eigenValues)).tolist()

        return eigenValues


    def getEigenVectors(self, real = True) -> List[List[int]]:
        """Menghasilkan basis ruang eigen dalam bentuk matrix dan sudah terurut menurut eigen value-nya."""
        eigenValues = self.getEigenValues()
        eigenVectors = []

        A = sp.Matrix(self.buffer)
        lamda = sp.Symbol("lamda", real=True)
        larr = sp.eye(len(self.buffer))*lamda
        B = larr-A

        temprow = []
        zeroMat = sp.zeros(B.rank(),1)
        for e in eigenValues:
            temp = B.copy().subs(lamda,e)
            sol, params = temp.gauss_jordan_solve(zeroMat)
            for param in params:
                taus = {tau:0 for tau in params}
                taus.update({param: 1})
                temprow = [tau[0] for tau in sol.xreplace(taus).tolist()]
                eigenVectors.append(temprow)

        eigenVectors = sp.Matrix(eigenVectors).T.tolist()
        return eigenVectors

# static method
def generateIdentityMatrix(dimension : int) -> List[List[int]] :
    # Membuat matriks identitas sesuai dimensi matriks A
    identity = [[0 for j in range(dimension)] for i in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            if (i == j):
                identity[i][j] = 1
    return identity
    

def QR_Decomposititon_GS(mat: List[List[int]]) -> tuple[List[List[int]], List[List[int]]]:
    """Dekomposisi QR dengan algoritma Gram-Schmidt Orthogonalization
    Too slow, still. O(2mn²).
    
    Must try Schwarz-Rutishauser Algorithm O(mn²)"""
    mat = np.array(mat)
    length = len(mat)

    e = np.empty((length,length))
    a = mat.T
    for i in range(length):
        u = np.copy(a[i])
        for j in range(i):
            u -= (a[i] @ e[j]) * e[j]
        e[i] = u/np.linalg.norm(u)

    R = np.zeros((length,length))
    for i in range(length):
        for j in range(i, length):
            R[i][j] = a[j]@e[i]
    Q = e.T

    return (Q,R)
