import cv2
import numpy as np
import src.utility as util

from src.eigen import getEigenVectors, getEigenValues
import time 

# will only be here in the development phase
def debugShow(name, mat):
    print('\t' + name)

    # as np float64
    mat = mat/mat.max()
    cv2.imshow(name, mat)

    # debug
    print(mat)
    print('mean = ' + str(np.sum(mat)/mat.shape[0]/mat.shape[1]) + '\n\n')
    print('type: ' , type(mat.flatten()[0]))


class EigenSolver():
    def __init__(self, desiredSize = None) -> None:
        self.hasTrained = False
        self.hasSolved = False

        self.desiredSize = desiredSize

        self.train_time = 0
        self.solve_time = 0

        self.other_results = []

    def train(self, files : np.ndarray, files_path : str) -> None :
        print('training started ...')
        startTime = time.time()

        imgCount = len(files)
        desiredSize = self.getDesiredSize()

        # make image files squared and centered
        # files = np.array([util.makeImageSquare(i) for i in files])

        # reformat files into usable images
        images = np.array([util.resize(i, desiredSize).flatten() for i in files]) # (x, 256^2)


        mean = np.mean([k for k in images], axis=0) # axis = 0 since we avg EVERY corresponding pixel
        

        imagesDiff = np.array([(images[i]-mean) for i in range(imgCount)])

        # A
        A = np.array([i for i in imagesDiff]).transpose() # sesuai definisi A di file
        

        # L = C'
        L =  A.transpose() @ A
        
        # compute eigenvalues and eigenvector of L
        eigValL, eigVecL = getEigenVectors(L, getEigenValues(L))
        

        # compute eigenvalues and eigenvector of C
        eigVecC = A @ eigVecL 
        
        # normalize eigenface
        eigVecC = util.normalizeSQR(eigVecC)

        # compute weights
        W = np.array([[np.array(eigVecC.transpose()[i]) @ np.array(imagesDiff[j].transpose()) for i in range(imgCount)] for j in range(imgCount)]) # ((eachW)eachGambar)

        # omega (array of weights distribution)
        Omega = np.array([i for i in W])
        
        # setting up values
        self.trainImgCount = imgCount
        self.mean = mean
        self.eigVec = eigVecC
        self.distributedWeight = Omega

        self.hasTrained = True
        self.files_path = files_path
        self.image_path = None


        print('training done')
        deltaTime = time.time() - startTime
        print('this process takes', deltaTime, 'seconds')

        self.train_time = f'{deltaTime:.2f}'

    
    def solve(self, new_files : np.ndarray, new_files_path : str) -> None:
        if not self.hasTrained:
            print("You haven't trained any image into the solver yet")
            return

        print('solving started ...')
        startTime = time.time()

        # new_files = np.array([util.makeImageSquare(i) for i in new_files]) # to make sure it is a square image

        desiredSize = self.getDesiredSize()
        mean = self.mean
        imgCount = self.trainImgCount
        eigVecC = self.eigVec
        
        newImgCount = len(new_files)

        newImages = np.array([util.resize(i, desiredSize).flatten() for i in new_files])
        newImagesDiff = np.array([(newImages[i]-mean) for i in range(newImgCount)])


        # kalkulasi pada targets
        W_target = np.array([[np.array(eigVecC.transpose()[i]) @ np.array(newImagesDiff[j].transpose()) for i in range(imgCount)] for j in range(newImgCount)]) # ((eachW)eachGambar)


        # omega
        Omega_target = np.array([i for i in W_target])


        # setting up values
        self.targetDistributedWeight = Omega_target
        self.targetImgCount = newImgCount

        self.hasSolved = True
        self.new_files_path = new_files_path

        print('solving done')
        deltaTime = time.time() - startTime
        print('this process takes', deltaTime, 'seconds')

        self.solve_time = f'{deltaTime:.2f}'

    def showResult(self):
        if not self.hasTrained or not self.hasSolved:
            print('please train images and solve for the solution before trying to show result')
            return

        # threshold_arr = [[util.getEuclidDistance(i, j) for i in self.distributedWeight] for j in self.distributedWeight]
        # threshold = np.max(np.array(threshold_arr).flatten())/2


        for i in range(self.targetImgCount):
            print('\n\nHasil pencocokan ke-' + str(i+1))
            j = 0
            minIdx = 0

            min = util.getEuclidDistance(self.targetDistributedWeight[i], self.distributedWeight[j])
            result = []
            while j < self.trainImgCount:
                
                if util.getEuclidDistance(self.targetDistributedWeight[i], self.distributedWeight[j]) < min:
                    min = util.getEuclidDistance(self.targetDistributedWeight[i], self.distributedWeight[j])
                    minIdx = j
                result.append(util.getEuclidDistance(self.targetDistributedWeight[i], self.distributedWeight[j]))
                j += 1
                
            
            print('Top 3 Result:')
            res = np.array(self.files_path)[np.argsort(result)][:3]
            for k in range(len(res)):
                print(res[k], ' \t with value: ', np.array(result)[np.argsort(result)][:3][k])
            
            print(self.files_path[minIdx], '<- path file training dengan kemiripan terbesar')
            print(self.new_files_path[i], '<- path file target pengenalan wajah')

            # for i in result:
            #     if i < threshold:
            #         print(i)

            self.image_path = self.files_path[minIdx]


            result_for_ui = np.array(self.files_path)[np.argsort(result)][:10]
            self.other_results = []
            for i in result_for_ui:
                n = i.split('/')[-1].split('\\')[-1].split('_')[0]
                if n not in self.other_results :
                    self.other_results.append(n)


        cv2.waitKey()



    # getter/setter
    def getDesiredSize(self) -> int:
        if self.desiredSize != None:
            return self.desiredSize
        else:
            return 256 # default