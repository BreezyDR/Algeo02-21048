import cv2
import numpy as np
import src.utility as util

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
        # files = np.array([np.array(i) for i in files])

        imgCount = len(files)
        desiredSize = self.getDesiredSize()

        # reformat files into usable images
        images = np.array([util.resize(i, desiredSize).flatten() for i in files]) # (x, 256^2)

        mean = np.mean([k for k in images], axis=0) # axis = 0 since we avg EVERY corresponding pixel

        # differ images
        imagesDiff = np.array([(images[i]-mean).astype(np.uint8) for i in range(imgCount)])

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
        # print('iegvecC',  eigVecC.shape)

        # compute weights
        W = np.array([[eigVecC.transpose()[i] @ imagesDiff[j] for i in range(imgCount)] for j in range(imgCount)])
        # print('2>>>>>>>>>>>>>>>>>.',  W.shape, W)


        # omega
        Omega = [W[i] @ eigVecC.transpose() for i in range(imgCount)] # somehow result nya kinda unsatisfying?
        # print(mean.shape, ' = ' ,(eigVecC).shape,  '+', (W[5]).shape , ' =' ,(eigVecC @ W[5].transpose()).shape, 'shapey')

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
        

        
        newImagesDiff = np.array([(newImages[i]-mean).astype(np.uint8) for i in range(newImgCount)])


        # for i in range(2, 6):
        #     debugShow('recon5' + str(i), util.unflatten(mean + eigVecC @ W[i].transpose()))

        # for i in range(len(eigVecC.transpose())):
        #     debugShow(str(i), util.unflatten(eigVecC.transpose()[i]))
        #     debugShow(str(i+1000), util.unflatten(Omega[i]))
        # debugShow('mean', util.unflatten(mean))

        #new stuffs
        W_target = np.array([[eigVecC.transpose()[i] @ newImagesDiff[j] for i in range(imgCount)] for j in range(newImgCount)])
        Omega_target = [W_target[i] @ eigVecC.transpose() for i in range(newImgCount)]


        # setting up values
        self.targetDistributedWeight = Omega_target
        self.targetImgCount = newImgCount

        self.hasSolved = True
        self.new_files_path = new_files_path


    def showResult(self):
        if not self.hasTrained or not self.hasSolved:
            print('please train images and solve for the solution before trying to show result')
            return

        for i in range(self.targetImgCount):
            print('\n\nHasil pencocokan ke-' + str(i+1))
            j = 0
            minIdx = 0

            min = np.linalg.norm(self.targetDistributedWeight[i] - self.distributedWeight[j])
            result = []
            while j < self.trainImgCount:
                # print(self.distributedWeight)
                if np.linalg.norm(self.targetDistributedWeight[i] - self.distributedWeight[j]) < min:
                    min = np.linalg.norm(self.targetDistributedWeight[i] - self.distributedWeight[j])
                    minIdx = j
                result.append(np.linalg.norm(self.targetDistributedWeight[i] - self.distributedWeight[j]))
                j += 1
                
            
            print('Top 3 Result:')
            res = np.array(self.files_path)[np.argsort(result)][:3]
            for k in range(len(res)):
                print(res[k], ' \t with value: ', np.array(result)[np.argsort(result)][:3][k])
            
            # print(np.mean(mean), np.mean(Omega[i]))

            # print(minIdx, 'min idx in mint')
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