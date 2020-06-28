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
import threading

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

from blueeyes.utils import Camera
from blueeyes.face_recognition import FaceRecognition, face_roi
from blueeyes.face_detection import FaceDetector
from blueeyes.tracking import Tracking

# mqtt for communication between modules
import paho.mqtt.client as mqtt

class EvidentWritter:
    def __init__(self):
        self.q = Queue()
        self.thread = threading.Thread(target=self._processing)
        self.state = 'run'
        self.thread.start()
    def _processing(self):
        while self.state == 'run' or not self.q.empty():
            frame, path = self.q.get_nowait()
            cv2.imwrite(path, frame)
    def push(self, frame, path):
        self.q.put((frame, path))
    def wait(self):
        self.state = 'stop'

# helper methods
def random_color():
    return tuple(np.random.choice(range(256), size=3))

def percent_of_majority(lst):
    d = {}
    for ele in lst:
        if ele not in d:
            d[ele] = 0
        d[ele] += 1 
    keys, values = list(zip(*d.items()))
    max_count = max(values)
    return keys[values.index(max_count)], max_count/len(lst)

def process_id(result_id, evident_path):
    recog_report = {
        'id': result_id.replace('\n', ''), 
        'time': time(),
        'evident_path': evident_path
    }
    mqtt_client.publish('/ids', json.dumps(recog_report))
    print(result_id)

HOME = os.environ['HOME']
EVIDENT_ROOT = str('../etc/evidence')

# Run MQTT client
mqtt_client = mqtt.Client()
# mqtt_client.connect('10.10.46.160', 1883, 60)
mqtt_client.connect('localhost', 1883, 60)
mqtt_client.loop_start()

cap = Camera(source='/home/huy/Downloads/3_nguoi_trucdien_DatHuyNhat.mp4', frameskip=1)
# cap = Camera(source='/home/huy/Downloads/Tung_nguoi_di_vao1.mp4', frameskip=0)
# cap = Camera(source='rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101', frameskip=5)
# cap = Camera(source=f'/home/huy/Downloads/{sys.argv[1]}.mp4', frameskip=0)
# cap = Camera(source='rtsp://Admin:12345@10.42.0.235:554/Streaming/Channels/101', frameskip=5)
# cap = Camera(source=0, frameskip=15)

# cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)

# Camera configurations
FRAME_HEIGHT = cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2
FRAME_WIDTH = cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
FRAME_COUNT_TO_DECIDE = 7

# start getting frame from video stream
cap.start()
# check video stream
if not cap.cap.isOpened():
    print('Error in opening camera')
    sys.exit(-1)

# Init core modules: detection, recoginition and tracking
detector = FaceDetector('mtcnn', min_face_size=20)
recog = FaceRecognition(
    feature_extractor_type='dlib',
    classifier_method='svm',
    model_path='/home/huy/face_recog/models/svm/rbf_c100_12062020_214524.svm'
    # model_path='/home/huy/face_recog/models/svm/rbf_c1000_12062020_181542.svm'
)
tracking = Tracking(deadline=FRAME_HEIGHT*0.7, threshold=0.475, max_live_time=10)

# evident writting thread
evident_writter = EvidentWritter()

# global variables used in while loop
frame_count = 0
color_table = {}
total_postprocessing = 0
execution_time = {}

while True:
    try: 
        # Read frame and frame available flag (ret)
        ret, frame = cap.read() 
        if ret:
            frame_count += 1
            frame = cv2.resize(frame, (0,0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)

            # detect face(s) in frame
            start_time = time()
            boxes = detector.detect(frame)
            execution_time['detection'] = time() - start_time
            start_time = time()

            # ignore too bright faces
            temp_boxes = []
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box
                    hsv = cv2.cvtColor(frame[y1:y2,x1:x2,:], cv2.COLOR_BGR2HSV)
                    brightness = cv2.mean(hsv)[2]
                    if 75 < brightness < 225:
                        temp_boxes.append(box)
                except:
                    print(box)
                    print(frame.shape)
                    input()

            # feature extraction from detected face(s)
            boxes = temp_boxes
            face_imgs = []
            for box in boxes:
                face_imgs.append(face_roi(frame, box))
            features = recog.extract_feature(face_imgs)

            # send feature(s) and face location to tracking module for later processing
            for i,box in enumerate(boxes):
                tracking.push([(box, features[i])])
            
            # post-processing with the support of tracking module
            pp_start = time()
            if frame_count % 5 == 0:
                tracking._check_buffer_status()
                print(tracking.count())                            
            i = 0 
            while i < tracking.count():
                if id(tracking.buffer[i]) not in color_table:
                    color = random_color()
                    color = tuple([int(x) for x in color])
                    color_table[id(tracking.buffer[i])] = color
                if len(tracking.box_history(i)) > 5:
                    box_to_draw = tracking.box_history(i)[-1]
                    print(box_to_draw)
                    if box_to_draw[0] > FRAME_WIDTH*0.8 or box_to_draw[0] < FRAME_WIDTH*0.2:
                        del(tracking.buffer[i])
                        i -= 1
                        continue 
                    detector.draw_bounding_box(frame, [box_to_draw], color_table[id(tracking.buffer[i])])
                    ### recognition phase
                    if frame_count % FRAME_COUNT_TO_DECIDE == 0:
                        features = tracking.features_history(i)
                        ids = recog.recog(features, threshold=0.4)
                        final_id, frequent = percent_of_majority(ids)
                        if frequent > 0.5:
                            time_str = datetime.now().strftime('%d-%m_%H-%M-%S')
                            evident_path = f'{EVIDENT_ROOT}/{final_id}_{time_str}.jpg'
                            process_id(final_id, evident_path)
                            evident_writter.push(frame, evident_path)
                i += 1
            total_postprocessing += time() - pp_start
            print(total_postprocessing/frame_count)
            
            # Show the frame to debug
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('Finalizing...')
        cap.stop()
        evident_writter.wait()
        break    

cv2.destroyAllWindows()

