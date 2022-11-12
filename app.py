import src.eigenfaces as ef
from src.Matrix import Matrix, generateIdentityMatrix

from src.featureExtraction import batch_extractor, show_img
from src.Matcher import Matcher

import cv2
import os

#debug
import random

if __name__ == '__main__':
    print(type('asd'))
    k = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    m = Matrix(k)
    
    print("determinan", m.getDeterminant())

    # ef.squashMat(k)

    m.describe()  # awal

    m.subtractBy(k)

    m.describe()  #setelah dikurangi dengan matrix dengan nilai yang sama

    print('\nadd images values matrix simulations\n')

    matOfMatrix = [Matrix() for i in range(10)]
    for i in range(10):
        dummyMat = [[random.randint(1, 10) for k in range(3)] for j in range(3)]  # buat n matrix random

        matOfMatrix[i].assign(dummyMat)

    summaryMat = Matrix(k)
    for i in range(10):
        summaryMat.addBy(matOfMatrix[i].getMatrix())
    
    summaryMat.divideBy(scalar=10)  # simulasi perata-rataan

    summaryMat.describe()

    toBeTransposed = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    toBeTransposed.describe()
    toBeTransposed.transpose()
    toBeTransposed.describe()


    # fE.batch_extractor('public/images/')
    # k = fE.extract_features('public/images/basuta.jpg')
    # print(len(k), len(k[0]))
    # fE.show_img('public/images/basuta.jpg')

    # l = cv2.imread('public/images/basuta.jpg')
    # l = cv2.resize(l, (l.shape[1]//4, l.shape[0]//4))
    # cv2.imshow('asdaf', l)
    # cv2.waitKey()
    
    images_path = 'public/images/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    batch_extractor(images_path)

    ma = Matcher('features.pck')

    for s in files:
        print('Query image ==========================================')
        # show_img(s)
        names, match = ma.match(s, topn=3)
        print('Result images ========================================')
        for i in range(3):
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            print('Match %s' % (1-match[i]))
            show_img(os.path.join(images_path, names[i]))