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

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('test'):
            face_image = face_recognition.load_image_file(f"test/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_encodings.append(face_encoding)
            self.known_names.append(image)
        print(self.known_names)
    
    def run_program(self):
        capturevideo = cv2.VideoCapture(0)

        if not capturevideo.isOpened():
            sys.exit('Video source not found!')

        while True:
            ret, frame = capturevideo.read()

            if self.is_current_frame:
                miniframe = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                miniframe_rgb = miniframe[:, :, ::-1]
                self.locations = face_recognition.face_locations(miniframe_rgb)
                self.encodings = face_recognition.face_encodings(miniframe_rgb, self.locations)

                self.names = []
                for encodings in self.encodings:
                    matching = face_recognition.compare_faces(self.known_encodings, encodings)
                    name = "Unknown"
                    confidence = '???'
                    distance = face_recognition.face_distance(self.known_encodings, encodings)
                    fitting_match = np.argmin(distance)
                    if matching[fitting_match]:
                        name = self.known_names[fitting_match]
                        confidence = face_limit(distance[fitting_match])
                    self.names.append(f'{name} ({confidence})')
            self.is_current_frame = not self.is_current_frame

            for (top, right, bottom, left), name in zip(self.locations, self.names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
            cv2.imshow('Face Recognition', frame)

            # Untuk berhenti merekam, klik 'Q' di keyboard
            if cv2.waitKey(1) == ord('q'):
                break
        capturevideo.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    webcam = FaceRecognition()
    webcam.run_program()


    
