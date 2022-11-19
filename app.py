import cv2
import numpy as np
import os

# from sklearn.preprocessing import normalize

import src.utility as util
from src.file import readFolder, readFile

if __name__ == '__main__':

    # read files streams
    files, files_path, imgCount = readFolder('public/images/')
    new_files, new_files_path, newImgCount = readFolder('public/target')

    # desired image size
    desiredSize = 256
    
    # reformat files into usable images
    images = np.array([util.resize(i, desiredSize).astype(np.uint8).flatten() for i in files])
    newImages = np.array([util.resize(i, desiredSize).astype(np.uint8).flatten() for i in new_files])


    # mean face
    mean = np.mean([k for k in images], axis=0).astype(np.uint8)

    # differ images
    imagesDiff = np.array([(images[i]-mean).astype(np.uint8) for i in range(imgCount)])
    newImagesDiff = np.array([(newImages[i]-mean).astype(np.uint8) for i in range(newImgCount)])


    # A
    A = np.array([i for i in images]).transpose() # sesuai definisi A di file

    # L = C'
    L =  A.transpose() @ A

    # compute eigenvalues and eigenvector of L
    eigValL, eigVecL = np.linalg.eig(L)

    # compute eigenvalues and eigenvector of C
    uAll = A @ eigVecL #eigVegU
    # uAll = normalize(uAll, axis=0, norm='l1') # 0 karena mereka (65535, 44) sehingga terbagi DI DALAM 65355 value itu, bukan 44
    uAll = util.normalizeNP(uAll)

    # compute weights
    W = np.array([[uAll.transpose()[i] @ imagesDiff[j] for i in range(imgCount)] for j in range(imgCount)])

    print(type(W[0][0]))
    print(util.normalizeNP(W), 's')

    # omega
    Omega = [W[i] @ uAll.transpose() for i in range(imgCount)]

    #new stuffs
    WssNew = np.array([[uAll.transpose()[i] @ newImagesDiff[j] for i in range(imgCount)] for j in range(newImgCount)])
    OmegassNew = [WssNew[i] @ uAll.transpose() for i in range(newImgCount)]

    for i in range(newImgCount):
        print('\n\nini ke - ' + str(i))
        xxx = 0
        minIdx = 0
        min = np.linalg.norm(OmegassNew[i] - Omega[xxx])
        while xxx < imgCount:
            if np.linalg.norm(OmegassNew[i] - Omega[xxx]) < min:
                min = np.linalg.norm(OmegassNew[i] - Omega[xxx])
                minIdx = xxx
            xxx += 1
        
        print(minIdx, 'min idx in mint')
        print(files_path[minIdx], 'ini path nya')
        print(new_files_path[i], 'ini path nya')
