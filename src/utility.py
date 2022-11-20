import cv2
import numpy as np

from sklearn.preprocessing import normalize


# pass check
def resize(arr : np.ndarray, size : tuple[int] | int) -> np.ndarray:
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

def magiclyNormalize(arr : np.ndarray) -> np.ndarray:
    print('from mag', arr.shape)
    return normalize([arr.transpose()], axis=1, norm='l1').transpose() # pls make sure it is a flatten mat

def magiclyDisplay(arr : np.ndarray, name : str = 'from utility.py') -> None:
    # it is a magic since we cant find why 25500 is the right spot
    cv2.imshow(name, arr*25500)


# normalize by comparison
def normalizeNP(arr : np.ndarray) -> np.ndarray:
    sum = np.sum(arr)

    mean = np.mean(arr)

    arr = np.array([[np.float64(abs(arr[j][i]/sum*100/mean)) for i in range(arr.shape[1])] for j in range(arr.shape[0])])

    return arr

# normalize by definition ||X|| = 1
def normalizeSQR(arr) -> np.ndarray:
    return normalize(arr) # pls make sure it is a flatten mat

# z = np.array([[-3, -1, 0 , 1], [-1, 1, 0 , -3]])
# print(normalize(z), normalizeSQR(z))