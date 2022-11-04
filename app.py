import src.eigenfaces as ef
from src.matrix import Matrix




if __name__ == '__main__':
    k = [[6,1,1], [4, -2, 5], [2,8,7]]
    m = Matrix(k)
    
    print("determinan: " + str(ef.dummy_getdet(k)))
    print("dete", m.dummy_getdet())

    ef.squashMat(k)

    m.describe()

    