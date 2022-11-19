import os, sys, cv2, math
import numpy as np
import face_recognition

def face_limit(dist, threshold=0.6):
    range = (1.0 - threshold)
    linear = (1.0 - dist) / (range * 2.0)

    if (dist > threshold):
        return str(round(linear * 100, 2)) + '%'
    else:
        nilai = (linear + ((1.0 - linear) * math.pow((linear - 0.5) * 2, 0.2))) * 100
        return str(round(nilai, 2)) + '%'

class FaceRecognition:
    locations = []
    encodings = []
    names = []
    known_encodings = []
    known_names = []
    is_current_frame = True
    
