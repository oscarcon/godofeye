import os
import sys
import cv2
from time import time, sleep
from queue import Queue
from collections import namedtuple

from facenet_pytorch import MTCNN, InceptionResnetV1

sys.path.append('../lib')
sys.path.append('../lib/yoloface')
from blueeyes.utils import Camera

cap = Camera(source='rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101', frameskip=5)
# cap = Camera(source=0, frameskip=0)
cap.start()

mtcnn = MTCNN(image_size=150, select_largest=False, keep_all=True, post_process=False, margin=40, device='cuda')

if not cap.cap.isOpened():
    print('Error in opening camera')

# cap = cv2.VideoCapture('rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101')

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, None, fx=1/2, fy=1/2)
        boxes, probs = mtcnn.detect(frame)
        print(boxes)
        if not isinstance(boxes, type(None)):
            for box in boxes:
                box = tuple(map(int, box))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,0,0), thickness=3)
        # if faces != None:
        #     for i, face in enumerate(faces):
        #         face_img = face.permute(1, 2, 0).int().numpy()
        #         face_img = face_img.astype('uint8')
        #         print(face_img.dtype)
        #         print(face_img)
        #         cv2.imshow(f"face{i}", face_img)
        cv2.imshow("cam", frame)
        if cv2.waitKey(1) == ord('q'):
            break
