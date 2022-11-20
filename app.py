import cv2
import numpy as np
import os

# from sklearn.preprocessing import normalize

import src.utility as util
from src.file import readFolder, readFile

# will only be here in the development phase
def debugShow(name, mat):
    print('\t' + name)

    # as int
    # cv2.imshow(name, mat.astype(np.uint8))

    # as np float64
    mat = mat/mat.max()
    cv2.imshow(name, mat)

    # debug
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
    images = np.array([util.resize(i, desiredSize).flatten() for i in files]) # (x, 256^2)
    newImages = np.array([util.resize(i, desiredSize).flatten() for i in new_files])


    # mean face
    mean = np.mean([k for k in images], axis=0) # axis = 0 since we avg EVERY corresponding pixel

    # differ images
    imagesDiff = np.array([(images[i]-mean).astype(np.uint8) for i in range(imgCount)])
    newImagesDiff = np.array([(newImages[i]-mean).astype(np.uint8) for i in range(newImgCount)])


    # A
    A = np.array([i for i in images]).transpose() # sesuai definisi A di file

    # L = C'
    L =  A.transpose() @ A

    # compute eigenvalues and eigenvector of L
    eigValL, eigVecL = np.linalg.eig(L)
    # eigVecL = util.normalizeSQR(eigVecL)

    # compute eigenvalues and eigenvector of C
    eigVecC = A @ eigVecL #eigVegU
    # eigVecC = util.normalizeNP(eigVecC)
    eigVecC = util.normalizeSQR(eigVecC)
    print('iegvecC',  eigVecC.shape)

    # compute weights
    W = np.array([[eigVecC.transpose()[i] @ imagesDiff[j] for i in range(imgCount)] for j in range(imgCount)])
    print('2>>>>>>>>>>>>>>>>>.',  W.shape, W)


    # omega
    Omega = [W[i] @ eigVecC.transpose() for i in range(imgCount)] # somehow result nya kinda unsatisfying?
    print(mean.shape, ' = ' ,(eigVecC).shape,  '+', (W[5]).shape , ' =' ,(eigVecC @ W[5].transpose()).shape, 'shapey')

    for i in range(2, 6):
        debugShow('recon5' + str(i), util.unflatten(mean + eigVecC @ W[i].transpose()))

    # for i in range(len(eigVecC.transpose())):
    #     debugShow(str(i), util.unflatten(eigVecC.transpose()[i]))
    #     debugShow(str(i+1000), util.unflatten(Omega[i]))
    debugShow('mean', util.unflatten(mean))

    #new stuffs
    W_target = np.array([[eigVecC.transpose()[i] @ newImagesDiff[j] for i in range(imgCount)] for j in range(newImgCount)])
    Omega_target = [W_target[i] @ eigVecC.transpose() for i in range(newImgCount)]

    for i in range(newImgCount):
        print('\n\nini ke - ' + str(i+1))
        j = 0
        minIdx = 0
        min = np.linalg.norm(Omega_target[i] - Omega[j])
        result = []
        while j < imgCount:
            if np.linalg.norm(Omega_target[i] - Omega[j]) < min:
                min = np.linalg.norm(Omega_target[i] - Omega[j])
                minIdx = j
            result.append(np.linalg.norm(Omega_target[i] - Omega[j]))
            j += 1
            
        
        print('Top 3 Result:')
        res = np.array(files_path)[np.argsort(result)][:3]
        for k in range(len(res)):
            print(res[k], ' \t with value: ', np.array(result)[np.argsort(result)][:3][k])
        
        # print(np.mean(mean), np.mean(Omega[i]))

        # print(minIdx, 'min idx in mint')
        print(files_path[minIdx], '<- path file training dengan kemiripan terbesar')
        print(new_files_path[i], '<- path file target pengenalan wajah')

    cv2.waitKey()