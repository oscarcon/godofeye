import cv2
import time
import logging
import numpy as np
import face_recognition
from mtcnn import MTCNN
logging.basicConfig(level=logging.DEBUG)

class FaceDetector:
    def __init__(self, type, **kwargs):
        self.type = type
        if self.type == 'haar':
            self.face_detector = cv2.CascadeClassifier('cascade_model/cascade_ignore_shirt.xml')
        elif self.type == 'mtcnn':
            self.face_detector = MTCNN(**kwargs)
    def detect(self, frame):
        '''
        frame: input image
        return: 
        '''
        if self.type == 'hog':
            boxes = face_recognition.face_locations(frame, number_of_times_to_upsample=1)
            # boxes format: css (top, right, bottom, left)
            boxes = [(y1, x1, y2, x2) for x1, y1, x2, y2 in boxes]
        elif self.type == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(15,15), flags=cv2.CASCADE_SCALE_IMAGE)
            boxes = [(x,y,x+w,y+h) for x,y,w,h in boxes]
        elif self.type == 'mtcnn':
            boxes = []
            for face in self.face_detector.detect_faces(frame):
                boxes.append(face['box'])
            boxes = [(x,y,x+w,y+h) for x,y,w,h in boxes]
        self.boxes = boxes
        return self.boxes
    def draw_bounding_box(self, frame, boxes):
        for box in boxes:
            if len(box) > 0:
                pt1  = (box[0], box[1])
                pt2 = (box[2], box[3])
                cv2.rectangle(frame, pt1, pt2, color=(0,0,255), thickness=2)
    def debug(self, frame):
        self.draw_bounding_box(frame, self.boxes)

if __name__ == '__main__':
    # cam = Camera()
    video = cv2.VideoCapture('person1.avi')
    video.set(cv2.CAP_PROP_FPS, 10)
    detector = Detector(type='haar')
    while True:
        try:
            ret, frame = video.read()
            detector.debug(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            break    
    cv2.destroyAllWindows()
    # img = cv2.imread("classe0.jpg", 1)
    # start = time.time()
    # for i in range(0,10):
    #     boxes = detector.detect(img)
    # runtime = time.time() - start
    # print(f'{runtime/10:.2f}s')

class HumanDetector:
    def __init__(self):
        pass
    def run(self, on_detect=None):
        # callback function on_detect when detected human incoming
        # args: (frame, n_human)
        pass