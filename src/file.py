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
    time = datetime.datetime.now()
    unique_id = str(time.year) + str(time.month) + str(time.day) + '_' + str(time.hour) + str(time.minute) + str(time.second)
    new_path = "./public/webcam/webcam_" + unique_id + '.jpg'
    cv2.imwrite(new_path, frame)

    return [frame], [new_path]



# dump calculated data into json
saves_path = "src/json/" # should be static
def readDataAsArray(exact_path: str) -> tuple[str, np.ndarray]:
    data = json.loads(os.path.join(saves_path, exact_path))

    return data["name"], np.ndarray(data["data"])

 # need optimization, preferrably go make it into int instead of float
def writeArrayAsData(exact_path: str, data: np.ndarray, name: str = 'written by writeArrayAsData') -> None:
    json_data = json.dumps({
                    "name" : name,
                    "data" : np.array(data).tolist()
                })

    with open(exact_path, 'w') as file:
        file.write(json_data)
    