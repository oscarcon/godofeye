import os
import sys
import cv2
import time
import logging
import numpy as np
import face_recognition

sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir)))

# set allow_growth for tensorflow
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras

logging.basicConfig(level=logging.DEBUG)


class FaceDetector:
    def __init__(self, type, scale=1, **kwargs):
        self.type = type
        self.scale = scale
        if self.type == 'yolo':
            from yolo.yolo import YOLO
            self.face_detector = YOLO(img_size=kwargs['model_img_size'])
        if self.type == 'haar':
            self.face_detector = cv2.CascadeClassifier('cascade_model/cascade_ignore_shirt.xml')
        elif self.type == 'mtcnn':
            from mtcnn import MTCNN
            kwargs['min_face_size'] //= self.scale
            self.min_face_size = kwargs['min_face_size']
            self.face_detector = MTCNN(**kwargs)
        elif self.type == 'mtcnn_torch':
            import mtcnn_torch as mtcnn
            pnet, rnet, onet = mtcnn.get_net_caffe('mtcnn_torch/model')
            self.face_detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')

    def detect(self, frame, size_ranges=[], brightness_ranges=[], filter=False):
        def is_in_size_ranges(box):
            left, top, right, bottom = box
            width = right - left
            height = bottom - top
            for wmin,hmin,wmax,hmax in size_ranges:
                if wmin <= width <= wmax and hmin <= height <= hmax:
                    return 
            return False
        def is_in_brightness_ranges(box):
            left, top, right, bottom = box
            for brightness_range in brightness_ranges:
                vmin,vmax = brightness_range 
                hsv = cv2.cvtColor(frame[top:bottom,left:right,:], cv2.COLOR_BGR2HSV)
                brightness = cv2.mean(hsv)[2]
                if vmin <= brightness <= vmax:
                    return True
            return False
        def zero_negative(box):
            return tuple([b if b > 0 else 0 for b in box])

        frame = cv2.resize(frame, (0,0), fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_CUBIC)

        if self.type == 'yolo':
            boxes = self.face_detector.detect(frame)
            boxes = [(y1,x1,y2,x2) for x1,y1,x2,y2 in boxes]
        elif self.type == 'hog':
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
        elif self.type == 'mtcnn_torch':
            boxes = []
            _boxes, landmarks = detector.detect(img)
            for box in _boxes:
                boxes.append(tuple([int(box[i]) for i in range(len(box))]))
        # zero negative value in box
        boxes = list(map(zero_negative, boxes))
        self.boxes = [tuple(map(lambda v: v*self.scale,box)) for box in boxes]

        if filter:
            boxes = [box for box in boxes if is_in_size_ranges(box)]
            boxes = [box for box in boxes if is_in_brightness_ranges(box)]
        return self.boxes

    def draw_bounding_box(self, frame, boxes, color):
        for box in boxes:
            if len(box) == 4:
                pt1  = (box[0], box[1])
                pt2 = (box[2], box[3])
                cv2.rectangle(frame, pt1, pt2, color, thickness=2)
        return frame
    def debug(self, frame):
        self.draw_bounding_box(frame, self.boxes, (255,255,0))

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