import src.eigenfaces as ef
from src.Matrix import Matrix, QR_Decomposititon_GS

# from src.featureExtraction import batch_extractor, show_img
# from src.Matcher import Matcher

import cv2
import numpy as np
import os

#debug
import random

if __name__ == '__main__':
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


    # zz1 = np.array([[1, 2, 4], [12, 13, 3], [13, 9, 3]])
    # u, l = QR_Decomposititon_GS(zz1)
    # print(zz1)
    # print(u)

    # AN ATTEMPT TO 100% USE NUMPY.ARRAY TO SOLVE FOR EVERYTHING
    # WE MIGHT MAKE AN numpy.array API WRAPPER INSTEAD IN THE FUTURE    

    # qq, qs = np.linalg.eig([[7, 3], [3, -1]])
    # print(qq, qs)   

    # sizeX, sizeY = 256, 256
    desiredSize = 256

    # cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
    images_path = 'public/images/'
    files_path = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    files = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in files_path] # modification-free files stream

    images = [Matrix(i) for i in files]

    for i in images:
        i.resize(desiredSize)

    print(images[0].shape)
    
    # images[9].cv2show()
    # cv2.imshow('bycv', files[9])

    # print(images[9].shape, np.array(files[9]).shape)
    













    # x, y = files[0].shape
    # print(x)
    # for i in range(len(files)):
    #     # files[i] = cv2.resize(files[i], (files[i].shape[1]//4, files[i].shape[0]//4)) # faster calculation
    #     files[i] = cv2.resize(files[i], (x, x), interpolation=cv2.INTER_CUBIC)
    #     files[i] = np.array(files[i]).reshape(-1, 1)
    #     print(files[i].shape)
        
    # filesMeaned = np.mean(files, axis=0).astype(np.uint8) # psi
    
    
    # filesDiff = [np.subtract(files[i], filesMeaned).astype(np.uint8) for i in range(len(files))] # a
    
    # fm = filesMeaned.reshape(int(filesMeaned.shape[0]**0.5), -1).astype(np.uint8)

    # for i in range(2):
    #     k = files[i].reshape(int((files[i].shape[0])**0.5), -1)
    #     kd = filesDiff[i].reshape(int((filesDiff[i].shape[0])**0.5), -1)
    #     cv2.imshow('asli' + str(i), k)
    #     cv2.imshow('diff' + str(i), kd)
    # # kdc = cv2.cvtColor(filesDiff[1], cv2.COLOR_BGR2RGB).reshape(int((filesDiff[1].shape[0])**0.5), -1)

    # # print(files[1].shape, filesDiff[1].shape)


    # c = np.multiply(files, np.transpose(files))
    # print(np.array(files).shape, c.shape)
    # # c' = a.at
    # # u = A.v
    # # c'u = l.u


    # eigwnp = np.linalg.eig(k)
    # print(eigwnp, len(eigwnp))

    # la, dmy, dmy2 = np.linalg.svd(k, full_matrices=0)
    # Q, R = QR_Decomposititon_GS(k)
    # print(Q, Q.shape)

    # cv2.imshow('with np', la)
    # cv2.imshow('result', Q*1000) #emang ga supposed to be shown
    # cv2.imshow('average', fm)
    # cv2.imshow('asli', k)
    # cv2.imshow('diff', kd)

    cv2.waitKey()
    

    



    # print(files)
    # print(np.mean(files, axis=0))
    
    # l = cv2.imread('public/images/basuta.jpg', cv2.IMREAD_GRAYSCALE)
    # l = cv2.resize(l, (l.shape[1]//4, l.shape[0]//4)) # faster calculation
    
    

    
    # print(np.array(l).shape)
    # cv2.imshow('result', l)
    # cv2.waitKey()
    
