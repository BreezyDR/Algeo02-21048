import cv2
import numpy as np
import os

# from sklearn.preprocessing import normalize

import src.utility as util
from src.file import readFolder, readFile

def debugShow(name, mat):
    print('\t' + name)

    # as int
    # cv2.imshow(name, mat.astype(np.uint8))

    # as np float64
    mat = mat/mat.max()
    cv2.imshow(name, mat)
    print(mat)
    print('mean = ' + str(np.sum(mat)/mat.shape[0]/mat.shape[1]) + '\n\n')
    print('type: ' , type(mat.flatten()[0]))

if __name__ == '__main__':

    # read files streams
    files, files_path, imgCount = readFolder('public/images/')
    new_files, new_files_path, newImgCount = readFolder('public/target')

    # desired image size
    desiredSize = 256
    
    # reformat files into usable images
    images = np.array([util.resize(i, desiredSize).flatten() for i in files])
    newImages = np.array([util.resize(i, desiredSize).flatten() for i in new_files])


    # mean face
    mean = np.mean([k for k in images], axis=0)

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

    # print(type(W[0][0]))
    # print(util.normalizeNP(W), 's')

    # omega
    Omega = [W[i] @ uAll.transpose() for i in range(imgCount)] # somehow result nya kinda unsatisfying?

    # for i in range(len(uAll.transpose())):
    #     debugShow(str(i), util.unflatten(uAll.transpose()[i]))
    #     debugShow(str(i+1000), util.unflatten(Omega[i]))
    # debugShow('mean', util.unflatten(mean))

    #new stuffs
    WssNew = np.array([[uAll.transpose()[i] @ newImagesDiff[j] for i in range(imgCount)] for j in range(newImgCount)])
    OmegassNew = [WssNew[i] @ uAll.transpose() for i in range(newImgCount)]

    for i in range(newImgCount):
        print('\n\nini ke - ' + str(i))
        xxx = 0
        minIdx = 0
        min = np.linalg.norm(OmegassNew[i] - Omega[xxx])
        result = []
        while xxx < imgCount:
            if np.linalg.norm(OmegassNew[i] - Omega[xxx]) < min:
                min = np.linalg.norm(OmegassNew[i] - Omega[xxx])
                minIdx = xxx
            result.append(np.linalg.norm(OmegassNew[i] - Omega[xxx]))
            xxx += 1
            
        
        print('Top 3 Result:')
        print(np.array(files_path)[np.argsort(result)][:3])
        # print('result: ', result)
        
        # debugShow('ori' + str(i), util.unflatten(np.array(images[i])))
        # debugShow('reconstruct' + str(i), util.unflatten(np.array(mean + Omega[i]*2000)))
        print(np.mean(mean), np.mean(Omega[i]))

        print(minIdx, 'min idx in mint')
        print(files_path[minIdx], 'ini path nya')
        print(new_files_path[i], 'ini path nya')

    cv2.waitKey()