import cv2
import numpy as np
import os

def readFolder(path: str) -> tuple[np.ndarray, str]:
    files_path = [os.path.join(path, p) for p in sorted(os.listdir(path))]
    files = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in files_path])

    return files, files_path

def readFile(exact_path: str) -> np.ndarray:
    file = cv2.imread(exact_path, cv2.IMREAD_GRAYSCALE)

    return file

