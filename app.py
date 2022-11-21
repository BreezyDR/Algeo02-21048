import cv2
import numpy as np
import os

# from sklearn.preprocessing import normalize

import src.utility as util
from src.file import readFolder, readFile

from src.EigenSolver import EigenSolver

if __name__ == '__main__':

    # read files streams
    files, files_path, imgCount = readFolder('public/images/')
    new_files, new_files_path, newImgCount = readFolder('public/target')

    # desired image size
    size = 256

    eigenSolver = EigenSolver(desiredSize=size)

    eigenSolver.train(files=files)
    eigenSolver.solve(new_files=new_files)

    Omega = eigenSolver.distributedWeight
    Omega_target = eigenSolver.targetDistributedWeight

    
    
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