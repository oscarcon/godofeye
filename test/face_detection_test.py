import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from time import time, sleep
from datetime import datetime
from queue import Queue
from collections import namedtuple
import pandas as pd
import threading
import multiprocessing
import signal
import subprocess
sys.path.append('../lib')

import tensorflow.compat.v1 as tf
import tensorflow.keras as keras

# allow gpu memory growth 
oldinit = tf.Session.__init__
def new_tfinit(session, target='', graph=None, config=None):
    print("Set config.gpu_options.allow_growth to True")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    oldinit(session, target, graph, config)
tf.Session.__init__ = new_tfinit

from blueeyes.face_detection import FaceDetector

SCALE = 2
detector = FaceDetector('faceboxes', min_face_size=70, scale=SCALE, threshold=0.5)

cap = cv2.VideoCapture('/home/blueeyes1/Downloads/hiv01288 00_00_20-00_00_30.mp4')

total_time = 0
frame_count = 0

while True:
    # Read frame and frame available flag (ret)
    ret, origin_frame = cap.read()
    if ret:
        # frame = cv2.normalize(origin_frame, None, 0, 255, cv2.NORM_MINMAX)
        # frame = cv2.GaussianBlur(origin_frame, (15,15), 0)
        # cv2.addWeighted(origin_frame, 1.7, frame, -0.5, 0, frame)
        frame = cv2.resize(origin_frame, (0,0), fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR)

        t = time()
        # detect face(s) in frame
        boxes = detector.detect(origin_frame)
        processing_t = time() - t
        print('FPS:', 1/processing_t)
        boxes = [tuple(map(lambda v: v//(SCALE), box)) for box in boxes]
        print(boxes)
        detector.draw_bounding_box(frame, boxes, (255,0,0))
        
        # Show the frame to debug
        cv2.imshow("frame", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break