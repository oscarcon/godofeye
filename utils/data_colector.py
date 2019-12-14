import os
import sys
import cv2
from sklearn.cluster import KMeans
import face_recognition

sys.path.append('../lib')

from blueeyes.face_recognition import Detector

cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
detector = Detector('mtcnn', min_face_size=50)

feature_vectors = []
labels = []

while True:
    try:
        ret, frame = cap.read()
        if ret:
            boxes = detector.detect(frame)
            for (x1, y1, x2, y2) in boxes:
                target_face = face_recognition.face_encodings(frame, known_face_locations=[(y1,x2,y2,x1)])[0]
                image_id = id(target_face)
                cv2.imwrite(f'face_images/{str(image_id)}.jpg', frame[y1:y2, x1:y2,:])
                feature_vectors.append(target_face)
                labels.append(labels)
                
            detector.debug(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        break    


cv2.destroyAllWindows()


