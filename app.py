import cv2
import numpy as np
import os

from sklearn.preprocessing import normalize

import src.utility as util
from src.file import readFolder, readFile

if __name__ == '__main__':
    files, files_path = readFolder('public/images/')

    new_file = readFile('public/target/target.jpg')

    desiredSize = 256
    

    images = np.array([util.resize(i, desiredSize).astype(np.uint8).flatten() for i in files])
    newImage = np.array(util.resize(new_file, desiredSize).astype(np.uint8).flatten())

    imgCount = len(images)


    # mean face
    mean = np.mean([k for k in images], axis=0).astype(np.uint8)

    # differ images
    imagesDiff = np.array([(images[i]-mean).astype(np.uint8) for i in range(imgCount)])
    newImageDiff = (newImage-mean).astype(np.uint8)

    # print('gaba', images.shape, mean.shape, imagesDiff.shape)
    # (44, 65536) (65536,) (44, 65536)

    # A
    A = np.array([i for i in images]).transpose() # sesuai definisi A di file
    print('asds', A.shape) # (65536, 44)

    # Covarianve Matrix
    # C = A @ A.transpose() / imgCount

    #instead

    # L = C'
    L =  A.transpose() @ A
    # (44, 44)

    # compute eigenvalues and eigenvector of L
    eigValL, eigVecL = np.linalg.eig(L)

    # print(eigValL.shape, eigVecL.shape)

    # compute eigenvalues and eigenvector of C
    uAll = A @ eigVecL #eigVegU
    uAll = normalize(uAll, axis=0, norm='l1') # 0 karena mereka (65535, 44) sehingga terbagi DI DALAM 65355 value itu, bukan 44

    # compute weights
    print('wei', uAll.transpose()[1].shape, imagesDiff[1].shape)
    W = np.array([[uAll.transpose()[i] @ imagesDiff[j] for i in range(imgCount)] for j in range(imgCount)])
    print(W.shape, 'was')
    print(W[1].shape, uAll.transpose()[1].shape)

    # omega
    Omega = [W[i] @ uAll.transpose() for i in range(imgCount)]

    #new stuffs
    WNew = [uAll.transpose()[i] @ newImageDiff for i in range(imgCount)]
    OmegaNew = WNew @ uAll.transpose()

    xxx = 0
    minIdx = 0
    min = np.linalg.norm(OmegaNew - Omega[xxx])
    while xxx < imgCount:
        if np.linalg.norm(OmegaNew - Omega[xxx]) < min:
            min = np.linalg.norm(OmegaNew - Omega[xxx])
            minIdx = xxx
        xxx += 1
    
    print(minIdx, 'min idx in mint')
    print(files_path[minIdx], 'ini path nya')


    # for i in range(4):
    #     powe1 = W[i] @ uAll.transpose()
    #     # print(mean.shape, powe1.shape, 'shapes')
    #     pic1 = (mean + (powe1 * 99 * 1000 / 13))/2
    #     # print('WOOOOO', mean, powe1, np.mean(mean), np.mean(powe1))
    #     pic1 =  pic1.reshape(int(pic1.shape[0]**.5), -1).astype(np.uint8)
        
        
        
    #     # print('sadss', pic1, pic1.shape)
    #     cv2.imshow('hito' + str(i), pic1)
    
    

    # kazz = uAll.transpose()[8] #* 25500 # why 25500?
    # print(max(kazz), 'max', 'mean:', int(np.mean(kazz*1000000)))
    # kazz = (kazz * 25500)
    # print(max(kazz), 'max')
    # print(np.sum(kazz), 'norval') #norm val
    # kazz = kazz.reshape(int(kazz.shape[0]**.5), -1)
    
    # print(kazz)

    # cv2.imshow('asdhash', kazz)
    # cv2.waitKey()
    
    
    # gb1 = np.array([eigVecL[1][i] * imagesDiff[i] for i in range(imgCount)])
    # print(gb1.shape)
    # ef1 = np.sum(gb1, axis=0).astype(np.uint8)
    # print(ef1.shape)

    # cv2.imshow('s',ef1)
    # cv2.waitKey()



    # cv2.imshow('nasa', mean)
    # for i in range(1, 4):
    #     cv2.imshow(str(i), imagesDiff[i])
    #     cv2.imshow(str(i+100), images[i])

    # cv2.waitKey()