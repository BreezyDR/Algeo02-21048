import src.eigenfaces as ef
from src.Matrix import Matrix, generateIdentityMatrix

# from src.featureExtraction import batch_extractor, show_img
# from src.Matcher import Matcher

import cv2
import numpy as np
import os

#debug
import random

if __name__ == '__main__':
    # print(type('asd'))
    # k = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    # m = Matrix(k)
    
    # print("determinan", m.getDeterminant())

    # # ef.squashMat(k)

    # m.describe()  # awal

    # m.subtractBy(k)

    # m.describe()  #setelah dikurangi dengan matrix dengan nilai yang sama

    # print('\nadd images values matrix simulations\n')

    # matOfMatrix = [Matrix() for i in range(10)]
    # for i in range(10):
    #     dummyMat = [[random.randint(1, 10) for k in range(3)] for j in range(3)]  # buat n matrix random

    #     matOfMatrix[i].assign(dummyMat)

    # summaryMat = Matrix(k)
    # for i in range(10):
    #     summaryMat.addBy(matOfMatrix[i].getMatrix())
    
    # summaryMat.divideBy(scalar=10)  # simulasi perata-rataan

    # summaryMat.describe()

    # toBeTransposed = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    # toBeTransposed.describe()
    # toBeTransposed.transpose()
    # toBeTransposed.describe()


    # fE.batch_extractor('public/images/')
    # k = fE.extract_features('public/images/basuta.jpg')
    # print(len(k), len(k[0]))
    # fE.show_img('public/images/basuta.jpg')



    # AN ATTEMPT TO 100% USE NUMPY.ARRAY TO SOLVE FOR EVERYTHING
    # WE MIGHT MAKE AN numpy.array API WRAPPER INSTEAD IN THE FUTURE

    # cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
    images_path = 'public/images/'
    files_path = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    files = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in files_path]
    
    for i in range(len(files)):
        files[i] = cv2.resize(files[i], (files[i].shape[1]//4, files[i].shape[0]//4)) # faster calculation
        files[i] = np.array(files[i]).reshape(-1, 1)
        
    filesMeaned = np.mean(files, axis=0) # psi
    
    filesDiff = [files[i] - filesMeaned for i in range(len(files))] # a

    



    # print(files)
    # print(np.mean(files, axis=0))
    
    # l = cv2.imread('public/images/basuta.jpg', cv2.IMREAD_GRAYSCALE)
    # l = cv2.resize(l, (l.shape[1]//4, l.shape[0]//4)) # faster calculation
    
    

    
    # print(np.array(l).shape)
    # cv2.imshow('result', l)
    # cv2.waitKey()
    
