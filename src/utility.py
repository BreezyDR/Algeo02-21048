import cv2
import numpy as np

from sklearn.preprocessing import normalize


# pass check
def resize(arr : np.ndarray, size : tuple[int] | int) -> np.ndarray:
    # print(arr)
    if not type(size) == int :
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_CUBIC) #here we use cv2.resize instead of np.resize,
                                                                  #since we need to interpolate missing values
    else :
        minDim = min(arr.shape[0], arr.shape[1])
        
        arr = cv2.resize(arr[:minDim, :minDim], (size, size), interpolation=cv2.INTER_CUBIC)
    
    return arr

# fixed
def unflatten(arr : np.ndarray) -> np.ndarray:
    # return resize(arr, (int(arr.shape[0]**.5), int(arr.shape[0]**.5)))
    return arr.reshape(int(arr.shape[0]**.5), -1)


# normalize by definition ||X|| = 1
def normalizeSQR(arr) -> np.ndarray:
    return normalize(arr) # pls make sure it is a flatten mat

def getEuclidDistance(om1, om2 = []): # om is a member list of omega
    sum = 0
    for i in range(len(om1)):
        if om2 != []:
            sum += (om1[i] - om2[i])**2
        else:
            sum += om1[i]**2

    return sum



def makeImageSquare(img: np.ndarray): # img should never be a flatten array
    if len(img.shape) > 2 :
        h, w, _ = img.shape
    else:
        h, w = img.shape

    if w > h:
        start_w = (w - h) // 2
        start_h = 0
    else:
        start_w = 0
        start_h = (h - 2) // 2

    return  cv2.resize(img[start_h:start_h+w, start_w:start_w+h], (min(w, h), min(w, h)))