import src.eigenfaces as ef
from src.matrix import Matrix, generateIdentityMatrix

#debug
import random

if __name__ == '__main__':
    k = [[6,1,1], [4, -2, 5], [2,8,7]]
    m = Matrix(k)
    
    print("determinan", m.getDeterminant())

    # ef.squashMat(k)

    m.describe() # awal

    m.subtractBy(k)

    m.describe() #setelah dikurangi dengan matrix dengan nilai yang sama

    print('\nadd images values matrix simulations\n')

    matOfMatrix = [Matrix() for i in range(10)]
    for i in range(10):
            dummyMat = [[random.randint(1,10) for i in range(3)] for j in range(3)] # buat n matrix random

            matOfMatrix[i].assign(dummyMat)

    summaryMat = Matrix(k)
    for i in range(10):
        summaryMat.addBy(matOfMatrix[i].getMatrix())
    
    summaryMat.divideBy(scalar=10) # simulasi perata-rataan

    summaryMat.describe()

    toBeTransposed = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    toBeTransposed.describe()
    toBeTransposed.transpose()
    toBeTransposed.describe()
    

    