import cv2
import numpy as np

from sklearn.preprocessing import normalize

def resize(arr : np.ndarray, size : tuple[int] | int) -> np.ndarray:
    if not type(size) == int :
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_CUBIC) #here we use cv2.resize instead of np.resize,
                                                                                #since we need to interpolate missing values
    else :
        minDim = min(arr.shape[0], arr.shape[1])
        
        arr = cv2.resize(arr[:minDim, :minDim], (size, size), interpolation=cv2.INTER_CUBIC)
    
    return arr

def unflatten(arr : np.ndarray) -> np.ndarray:
    return resize(arr, (int(arr.shape[0]**.5), int(arr.shape[0]**.5)))

def magiclyNormalize(arr : np.ndarray) -> np.ndarray:
    print('from mag', arr.shape)
    return normalize([arr.transpose()], axis=1, norm='l1').transpose() # pls make sure it is a flatten mat

def magiclyDisplay(arr : np.ndarray, name : str = 'from utility.py') -> None:
    # it is a magic since we cant find why 25500 is the right spot
    cv2.imshow(name, arr*25500)

def normalizeNP(arr : np.ndarray) -> np.ndarray:
    sum = np.sum(arr)

    print(arr.shape)

    arr = np.array([[np.float64(arr[j][i]/sum) for i in range(arr.shape[1])] for j in range(arr.shape[0])])

    return arr