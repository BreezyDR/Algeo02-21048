from typing import List, Optional
from numpy.typing import ArrayLike

import numpy as np
import sympy as sp
from tabulate import tabulate
from scipy.linalg import hessenberg
import cv2


class Matrix:
    def __init__(self, matrix : ArrayLike = None) -> None:
        if np.any(matrix) != None:
            self.assign(matrix=matrix)

    def __call__(self) -> np.ndarray :
        return self.buffer

    def __str__(self) -> str(np.ndarray):
        return str(self.buffer)

    def __repr__(self) -> str(np.ndarray):
        return str(self.buffer)
    
    def __add__(self, p) :
        return Matrix(np.add(self.buffer, p.buffer))

    def __sub__(self, p) :
        return Matrix(np.subtract(self.buffer, p.buffer))

    def __mul__(self, p) :
        return Matrix(np.multiply(self.buffer, p.buffer))
    
    def __matmul__(self, p) :
        return Matrix(np.matmul(self.buffer, p.buffer))

    def __iadd__(self, p) :
        self.addBy(p.buffer)
        return self

    def __isub__(self, p) :
        self.subtractBy(p.buffer)
        return self

    def __imul__(self, p) :
        self.multiplyBy(p.buffer)
        return self

    def __idiv__(self, p) :
        self.divideBy(p.buffer)
        return self

    def __imulmat__(self, p) :
        self.buffer @= p.buffer
        return self


    def assign(self, matrix: ArrayLike) :
        self.buffer = np.array(matrix).astype(np.uint8) # typecast here might causes issues

        if len(self.buffer.shape) <= 1 :
            self.shape = (self.buffer.shape[0], 1)
        else:
            self.shape = self.buffer.shape

        self.T = self.buffer

        self.adjustSize()

        return self

    def adjustSize(self) -> None :
        self.sizeX = self.shape[0]
        self.sizeY = self.shape[1]

    def addBy(self, matrix : ArrayLike = None) -> None :
        if matrix is None:
            return

        matrix = np.array(matrix)

        if self.buffer.shape != matrix.shape :
            print('IN Matrix.py, ADDED 2 ARRAYS OF DIFFERENT SHAPE')
        
        self.assign(np.add(self.buffer, matrix))

        return self

    def subtractBy(self, matrix : ArrayLike = None) -> None :
        if matrix is None:
            return

        matrix = np.array(matrix)

        if self.buffer.shape != matrix.shape :
            print('IN Matrix.py, SUBTRACTED 2 ARRAYS OF DIFFERENT SHAPE')

        self.assign(np.subtract(self.buffer, matrix))

        return self

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

        return self


    def divideBy(self, scalar : int = None) -> None :
        if scalar != 0 and scalar != None:
            self.assign(np.divide(self.buffer, scalar))

        return self

    def resize(self, size : tuple[int] | int) -> None:
        if not type(size) == int :
            self.assign(cv2.resize(self.buffer, size, interpolation=cv2.INTER_CUBIC)) #here we use cv2.resize instead of np.resize,
                                                                                    #since we need to interpolate missing values
        else :
            minDim = min(self.sizeX, self.sizeY)
            
            self.assign(cv2.resize(self.buffer[:minDim, :minDim], (size, size), interpolation=cv2.INTER_CUBIC))
        
        return self

        
    # linear algebra operational
    def transpose(self) -> None :
        self.assign(np.transpose(self.buffer))

        return self

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

        return self

    def cv2show(self, name : str = 'default') -> None:
        cv2.imshow(name, self.buffer)

    #getter setter
    def getSizeX(self) -> int :
        return self.sizeX

    def getSizeY(self) -> int :
        return self.sizeY

    def getMatrix(self) -> np.ndarray :
        return self.buffer

    def getSquashedMatrix(self) -> np.ndarray :
        return self.buffer.reshape(-1, 1)

    def getUnsquashedMatrix(self) -> np.ndarray :
        return self.buffer.reshape(-1, int(np.ceil(self.shape[0]**.5)))
    
    def getTransposedMatrix(self) -> np.ndarray :
        return self.buffer.transpose()

    def isSquare(self) -> bool:
        return (self.buffer.shape[0] == self.buffer.shape[1])

    def getShape(self) -> tuple[int]:
        return self.buffer.shape
    
    
    @staticmethod
    def qr_decomposititon_gs(mat: List[List[int]]) -> tuple[List[List[int]], List[List[int]]]:
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
                u = u - (a[i] @ e[j]) * e[j]
            e[i] = u/np.linalg.norm(u)

        R = np.zeros((length,length))
        for i in range(length):
            for j in range(i, length):
                R[i][j] = a[j]@e[i]
        Q = e.T

        return (Q,R)

    
    @staticmethod
    def qr_gs_modsr(A):
        A = np.array(A)
        (m,n) = np.shape(A)

        Q = np.array(A)      
        R = np.zeros((n, n))

        for k in range(n):
            for i in range(k):
                R[i,k] = Q[:,i].T@(Q[:,k])
                Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]

            R[k,k] = np.linalg.norm(Q[:,k]); Q[:,k] = Q[:,k] / R[k,k]
        
        return -Q, -R

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
            # u -= (a[i] @ e[j]) * e[j]
            u = u - (a[i] @ e[j]) * e[j] # note: ntah mengapa harus diginiin, karena bakal error otherwise
        e[i] = u/np.linalg.norm(u)

    R = np.zeros((length,length))
    for i in range(length):
        for j in range(i, length):
            R[i][j] = a[j]@e[i]
    Q = e.T

    return (Q,R)
