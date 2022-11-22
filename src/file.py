import cv2
import numpy as np
import os

import json
import datetime

# read image from files
def readFolder(path: str) -> tuple[np.ndarray, str, int]:
    files_path = [os.path.join(path, p) for p in sorted(os.listdir(path))]
    files = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in files_path])

    return files, files_path

def readFile(exact_path: str) -> np.ndarray:
    file = np.array([cv2.imread(exact_path, cv2.IMREAD_GRAYSCALE)])

    return file, [exact_path]

def readWebCam(frame: np.ndarray) :
    unique_id = getUniqueId()
    new_path = "/public/webcam/webcam_" + unique_id + '.jpg'
    new_path =  os.path.join(os.getcwd(), new_path)
    cv2.imwrite(new_path, frame)

    return [frame], [new_path]



# dump calculated data into json
def readDataAsArray(absolute_path: str) -> tuple[int, np.array, np.array, np.array]:
    with open(absolute_path, 'r') as file:
        data = json.loads(file.read())

    return data["img_count"], np.array(data["mean_face"]), np.array(data["eigen_vectors"]), np.array(data["distributed_weight"]), np.array(data["files_path"])

 # need optimization, preferrably go make it into int instead of float
def writeArrayAsData(absolute_path: str, img_count : int, mean_face : np.ndarray, eigen_vectors: np.ndarray, distributed_weight: np.ndarray, files_path: np.ndarray) -> None:
    json_data = json.dumps({
                    "img_count": img_count,
                    "mean_face": np.array(mean_face).tolist(),
                    "eigen_vectors" : np.array(eigen_vectors).tolist(),
                    "distributed_weight": np.array(distributed_weight).tolist(),
                    "files_path": np.array(files_path).tolist()
                })
    
    with open(absolute_path, 'w') as file:
        file.write(json_data)


# helper to naming

def getUniqueId() -> str:
    time = datetime.datetime.now()
    unique_id = str(time.year) + str(time.month) + str(time.day) + '_' + str(time.hour) + str(time.minute) + str(time.second)

    return unique_id