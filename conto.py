import cv2
import numpy as np
import os

def createNewFace(*args):
    # Start with the mean image
    output = averageFace

    # Add the eigen faces with the weights
    for i in range(0, NUM_EIGEN_FACES):
        '''
        OpenCV does not allow slider values to be negative. 
        So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
        '''
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sliderValues[i] - MAX_SLIDER_VALUE/2
        output = np.add(output, eigenFaces[i] * weight)

    # Display Result at 2x size
    output = cv2.resize(output, (0,0), fx=2, fy=2)
    cv2.imshow("Result", output)

def createDataMatrix(images):
    print("Creating data matrix",end=" ... ")
    ''' 
    Allocate space for all images in one data matrix.
    The size of the data matrix is
    ( w  * h  * 3, numImages )
    where,
    w = width of an image in the dataset.
    h = height of an image in the dataset.
    3 is for the 3 color channels.
    '''

    numImages = len(images)
    sz = images[0].shape
    print(sz, 'awda')
    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        data[i,:] = image

    print("DONE")
    return data

def readImages(images_path):
    files_path = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    files = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in files_path] # modification-fre

    return files


if __name__ == '__main__': 
    # Number of EigenFaces
    NUM_EIGEN_FACES = 10

    # Maximum weight
    MAX_SLIDER_VALUE = 255

    # Directory containing images
    dirName = 'public/images/'

    # Read images
    images = readImages(dirName)

    # Size of images
    sz = images[0].shape

    # Create data matrix for PCA.
    data = createDataMatrix(images)

    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print ("DONE")

    averageFace = mean.reshape(sz)

    eigenFaces = []; 

    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)

    # Create window for displaying Mean Face
    cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

    # Display result at 2x size
    output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
    cv2.imshow("Result", output)

    # Create Window for trackbars
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)

    sliderValues = []


    print('''Usage:
    Change the weights using the sliders
    Click on the result window to reset sliders
    Hit ESC to terminate program.''')

    cv2.waitKey(0)
    cv2.destroyAllWindows()