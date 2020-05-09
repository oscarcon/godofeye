import os
import sys
import cv2
from time import time, sleep
from queue import Queue
from collections import namedtuple

sys.path.append('../lib')
sys.path.append('../lib/yoloface')
from blueeyes.utils import Camera

cap = Camera(source='rtsp://Admin:@10.42.0.235:554', frameskip=5)

cap.start()

if not cap.cap.isOpened():
    print('Error in opening camera')

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("cam", frame)
        if cv2.waitKey(0) == ord('q'):
            break
