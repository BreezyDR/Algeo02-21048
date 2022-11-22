import cv2
import numpy as np
import src.utility as util

from src.eigen import getEigenVectors, getEigenValues

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


class EigenSolver():
    def __init__(self, desiredSize = None) -> None:
        self.hasTrained = False
        self.hasSolved = False

        self.desiredSize = desiredSize

    def train(self, files : str, files_path : str) -> None :
        imgCount = len(files)
        desiredSize = self.getDesiredSize()

        # reformat files into usable images
        images = np.array([util.resize(i, desiredSize).flatten() for i in files]) # (x, 256^2)

        # data = np.array([[[1, 1], [-2, -3]], [[1, -1], [3, 2]], [[2, -2], [1, 3]], [[1, 2], [2, 1]]])
        # images = np.array([i.transpose().flatten() for i in data])
        # print(images)

        mean = np.mean([k for k in images], axis=0) # axis = 0 since we avg EVERY corresponding pixel
        # print(mean)

        # differ images
        imagesDiff = np.array([(images[i]-mean) for i in range(imgCount)])
        # print(imagesDiff)

                # A
        A = np.array([i for i in imagesDiff]).transpose() # sesuai definisi A di file
        # print(A, 'a')

        # L = C'
        L =  A.transpose() @ A
        # dum = (A @ A.transpose())/4
        # print(dum)
        # print(dum/4)
        # compute eigenvalues and eigenvector of L
        eigValL, eigVecL = np.linalg.eig(L)
        # tempL = getEigenValues(L)
        # eigValL, eigVecL = getEigenVectors(L, tempL)
        # print(eigValL, eigVecL)
        # print(util.normalizeSQR(eigVecL), 'norm')
        

        # compute eigenvalues and eigenvector of C
        eigVecC = A @ eigVecL #eigVegU
        # print(eigVecC)
        
        eigVecC = util.normalizeSQR(eigVecC)


        # compute weights
        # W = np.array([[eigVecC.transpose()[i] @ imagesDiff[j] for i in range(imgCount)] for j in range(imgCount)])
        W = np.array([[np.array(eigVecC.transpose()[i]) @ np.array(imagesDiff[j].transpose()) for i in range(imgCount)] for j in range(imgCount)]) # ((eachW)eachGambar)
        # print('shap', np.array(W).shape, np.array(eigVecC.transpose()[0]).shape, np.array(imagesDiff[0].transpose()).shape)


        # omega
        # Omega = [W[i] @ eigVecC.transpose() for i in range(imgCount)] # somehow result nya kinda unsatisfying?
        Omega = np.array([i for i in W])

        # print('here',np.array(mean + eigVecC @ Omega[1].transpose()).shape)
        # debugShow('me', util.unflatten(mean))
        # for i in range(imgCount):
        #     debugShow('fave' + str(i), util.unflatten(mean + eigVecC @ Omega[i].transpose()))
        

        # cv2.waitKey()

        

        # setting up values
        self.trainImgCount = imgCount
        self.mean = mean
        self.eigVec = eigVecC
        self.distributedWeight = Omega

        self.hasTrained = True
        self.files_path = files_path
        self.image_path = None

    
    def solve(self, new_files : str, new_files_path : str) -> None:
        if not self.hasTrained:
            print("You haven't trained any image into the solver yet")
            return

        new_files = np.array(new_files)

        desiredSize = self.getDesiredSize()
        mean = self.mean
        imgCount = self.trainImgCount
        eigVecC = self.eigVec
        
        newImgCount = len(new_files)

        
        newImages = np.array([util.resize(i, desiredSize).flatten() for i in new_files])


        # mean face
        

        
        newImagesDiff = np.array([(newImages[i]-mean) for i in range(newImgCount)])


        # kalkulasi pada targets
        W_target = np.array([[np.array(eigVecC.transpose()[i]) @ np.array(newImagesDiff[j].transpose()) for i in range(imgCount)] for j in range(newImgCount)]) # ((eachW)eachGambar)
        # print('shap', np.array(W).shape, np.array(eigVecC.transpose()[0]).shape, np.array(imagesDiff[0].transpose()).shape)


        # omega
        # Omega = [W[i] @ eigVecC.transpose() for i in range(imgCount)] # somehow result nya kinda unsatisfying?
        Omega_target = np.array([i for i in W_target])


        # setting up values
        self.targetDistributedWeight = Omega_target
        self.targetImgCount = newImgCount

        self.hasSolved = True
        self.new_files_path = new_files_path

    def getEuclidDistance(self, om1, om2): #member of omega
        sum = 0
        for i in range(len(om1)):
            sum += (om1[i] - om2[i])**2

        return sum



    def showResult(self):
        if not self.hasTrained or not self.hasSolved:
            print('please train images and solve for the solution before trying to show result')
            return

        for i in range(self.targetImgCount):
            print('\n\nHasil pencocokan ke-' + str(i+1))
            j = 0
            minIdx = 0

            # min = np.linalg.norm(self.targetDistributedWeight[i] - self.distributedWeight[j])
            min = self.getEuclidDistance(self.targetDistributedWeight[i], self.distributedWeight[j])
            result = []
            while j < self.trainImgCount:
                
                if self.getEuclidDistance(self.targetDistributedWeight[i], self.distributedWeight[j]) < min:
                # if np.linalg.norm(self.targetDistributedWeight[i] - self.distributedWeight[j]) < min:
                    # min = np.linalg.norm(self.targetDistributedWeight[i] - self.distributedWeight[j])
                    min = self.getEuclidDistance(self.targetDistributedWeight[i], self.distributedWeight[j])
                    minIdx = j
                # result.append(np.linalg.norm(self.targetDistributedWeight[i] - self.distributedWeight[j]))
                result.append(self.getEuclidDistance(self.targetDistributedWeight[i], self.distributedWeight[j]))
                j += 1
                
            
            print('Top 3 Result:')
            res = np.array(self.files_path)[np.argsort(result)][:self.targetImgCount]
            for k in range(len(res)):
                print(res[k], ' \t with value: ', np.array(result)[np.argsort(result)][:self.targetImgCount][k])
            
            print(self.files_path[minIdx], '<- path file training dengan kemiripan terbesar')
            print(self.new_files_path[i], '<- path file target pengenalan wajah')

            self.image_path = self.files_path[minIdx]

        cv2.waitKey()



    # getter/setter
    def getDesiredSize(self) -> int:
        if self.desiredSize != None:
            return self.desiredSize
        else:
            return 256 # default