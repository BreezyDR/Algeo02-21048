import src.eigenfaces as ef
from src.matrix import Matrix, identity_matrix

#debug
import random

if __name__ == '__main__':
    k = [[6,1,1], [4, -2, 5], [2,8,7]]
    m = Matrix(k)
    
    print("determinan", m.dummy_getdet())

    # ef.squashMat(k)

    m.describe()

    m.subtractBy(k)

    m.describe()

    print('\nadd images values matrix simulations\n')

    matOfMatrix = [Matrix() for i in range(10)]
    for i in range(10):
            dummyMat = [[random.randint(1,10) for i in range(3)] for j in range(3)]

            matOfMatrix[i].assign(dummyMat)

    summaryMat = Matrix(k)
    for i in range(10):
        summaryMat.addBy(matOfMatrix[i].getMatrix())
    
    summaryMat.divideBy(scalar=10)

    summaryMat.describe()

    toBeTransposed = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    toBeTransposed.describe()
    toBeTransposed.transpose()
    toBeTransposed.describe()
    

    