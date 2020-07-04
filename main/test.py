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

if len(sys.argv) < 2:
    print('Not enough argument for script')
    sys.exit(-1)

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

from blueeyes.config import *
from blueeyes.utils import Camera
from blueeyes.face_recognition import FaceRecognition, face_roi
from blueeyes.face_detection import FaceDetector
from blueeyes.tracking import Tracking
from blueeyes.utils import WDT

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
            frame, path = self.q.get()
            cv2.imwrite(path, frame)
    def push(self, frame, path):
        self.q.put((frame, path))
    def wait(self):
        self.state = 'stop'

# class VideoWritterProcess(multiprocessing.Process):
#     def __init__(self):
#         super(VideoWritterProcess, self).__init__()
#         self.cap = cv2.VideoCapture('/home/huy/Downloads/3_nguoi_trucdien_DatHuyNhat.mp4')
#         self.video_writter = cv2.VideoWriter(f'video{sys.argv[1]}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (800,600))
#         self.state = 'run'
#         signal.signal(signal.SIGUSR1, self._signal_handler)
#     def _signal_handler(self, signum, frame):
#         print('VideoWritter received signal')
#         sys.stdout.flush()
#         if signum == signal.SIGUSR1:
#             self.state = 'stop'
#     def run(self):
#         while self.state == 'run':
#             ret, frame = self.cap.read()
#             if ret:
#                 frame = cv2.resize(frame, (800,600))
#                 self.video_writter.write(frame)
#         print('Writer exited')
#         self.video_writter.release()
#         self.cap.release()

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

# Run MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.loop_start()

cap_source = 'rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101'
cap = Camera(source=cap_source, frameskip=10)
# cap = Camera(source='/home/huy/Downloads/3_nguoi_trucdien_DatHuyNhat.mp4', frameskip=1)
# cap = Camera(source='/home/huy/Downloads/Tung_nguoi_di_vao1.mp4', frameskip=0)
# cap = Camera(source='rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101', frameskip=5)
# cap = Camera(source=f'/home/huy/Downloads/{sys.argv[1]}.mp4', frameskip=0)
# cap = Camera(source='rtsp://Admin:12345@10.42.0.235:554/Streaming/Channels/101', frameskip=5)
# cap = Camera(source=0, frameskip=15)

# cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)

# Camera configurations
FRAME_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
FRAME_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# start getting frame from video stream
cap.start()

# check video stream
if not cap.isOpened():
    print('Error in opening camera')
    sys.exit(-1)

# Init core modules: detection, recoginition and tracking
SCALE = 2
detector = FaceDetector('faceboxes', min_face_size=70, scale=SCALE, threshold=0.3)
recog = FaceRecognition(
    feature_extractor_type='dlib',
    classifier_method='svm',
    model_path='/home/huy/face_recog/models/svm/rbf_c100_12062020_214524.svm'
    # model_path='/home/huy/face_recog/models/svm/rbf_c1000_12062020_181542.svm'
)
tracking = Tracking(deadline=FRAME_HEIGHT*0.7, threshold=0.475, max_live_time=3)

# evident writting thread
evident_writter = EvidentWritter()
# video_writter = VideoWritterProcess()
# video_writter.start()
# video_writter_process = subprocess.Popen(['/home/huy/venv/cv/bin/python', 'recorder.py', cap_source, str(sys.argv[1])])

# Reset when the system is hanging
wdt = WDT(timeout=60)

# recog report
records = []

# pd.DataFrame(columns=['Time', 'Source', 'ID', 'Path'])
# global variables used in while loop
frame_count = 0
color_table = {}
total_postprocessing = 0
execution_time = {}

while True:
    try: 
        # Read frame and frame available flag (ret)
        ret, origin_frame = cap.read()
        if ret:
            # Notify the watchdog timer
            wdt.notify()

            start_time = time()
            print('Got frame')
            frame_count += 1
            frame = cv2.resize(origin_frame, (0,0), fx=1/2,fy=1/2, interpolation=cv2.INTER_LINEAR)

            # detect face(s) in frame
            boxes = detector.detect(origin_frame)
            execution_time['detection'] = time() - start_time
            
            # ignore too bright faces
            temp_boxes = []
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box
                    hsv = cv2.cvtColor(origin_frame[y1:y2,x1:x2,:], cv2.COLOR_BGR2HSV)
                    brightness = cv2.mean(hsv)[2]
                    if 75 < brightness < 225:
                        temp_boxes.append(box)
                except:
                    print(box)
                    print(origin_frame.shape)
                    # input()
            
            t_temp = time()
            # feature extraction from detected face(s)
            boxes = temp_boxes
            face_imgs = []
            for box in boxes:
                face_imgs.append(face_roi(origin_frame, box))
            features = recog.extract_feature(face_imgs)
            execution_time['extraction'] = time() - t_temp

            # send features face location to tracking module for later processing
            for i,box in enumerate(boxes):
                tracking.push([(box, features[i])])
            
            t_temp = time()
            # predict every frame
            if features:
                predict_ids = recog.recog(features, threshold=0.4)
                t = datetime.now().strftime('%d/%m %H:%M:%S')
                for i, predict_id in enumerate(predict_ids):
                    record = [t, 'frame', str(boxes[i]), predict_id, '']
                    records.append(record)
                    with open(f'report{sys.argv[1]}.csv', 'a+', buffering=1) as report:
                        report.write('\t'.join(record) + '\n')
                        report.flush()
            execution_time['recognition'] = time() - t_temp

            # post-processing with the support of tracking module
            t_temp = time()
            if frame_count % 5 == 0:
                tracking._check_buffer_status()
                print(tracking.count())                            
            i = 0 
            while i < tracking.count():
                if id(tracking.buffer[i]) not in color_table:
                    color = random_color()
                    color = tuple([int(x) for x in color])
                    color_table[id(tracking.buffer[i])] = color
                box_to_draw = tracking.box_history(i)[-1]
                box_to_draw =  tuple(map(lambda v: v//2,box_to_draw)) 
                print(box_to_draw)
                detector.draw_bounding_box(frame, [box_to_draw], color_table[id(tracking.buffer[i])])
                if box_to_draw[0] > FRAME_WIDTH/2*0.8 or box_to_draw[0] < FRAME_WIDTH/2*0.2:
                    del(tracking.buffer[i])
                    if i != 0:
                        i -= 1
                    continue 
                if len(tracking.box_history(i)) >= FRAME_COUNT_TO_DECIDE:
                    ### recognition phase
                    features = tracking.features_history(i)
                    ids = recog.recog(features, threshold=0.4)
                    final_id, frequent = percent_of_majority(ids)
                    if frequent > 0.5:
                        t = datetime.now().strftime('%d/%m %H:%M:%S')
                        time_str = datetime.now().strftime('%d-%m_%H-%M-%S')
                        if final_id == 'unknown':
                            sub_path = 'sbuilding/unknown'
                        else:
                            sub_path = 'subilding/known'
                        evident_path = f'{EVIDENT_ROOT}/{sub_path}/{final_id}_{time_str}.jpg'
                        process_id(final_id, f'{sub_path}/{final_id}_{time_str}.jpg')
                        evident_writter.push(frame, evident_path)
                        record = [t, 'postprocessing', str(tracking.box_history(i)[-1]), final_id, evident_path]
                        records.append(record)
                        with open(f'report{sys.argv[1]}.csv', 'a+', buffering=1) as report:
                            report.write('\t'.join(record) + '\n')
                            report.flush()
                        # # Remove the buffer of processed person
                        # del(tracking.buffer[i])
                        # if i != 0:
                        #     i -= 1
                i += 1
            execution_time['postprocessing'] = time() - t_temp
            execution_time['total'] = time() - start_time
            
            print(execution_time)
            
            # Show the frame to debug
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('Finalizing...')
        cap.stop()
        evident_writter.wait()
        video_writter_process.send_signal(signal.SIGUSR1)
        video_writter_process.communicate()
        # os.kill(video_writter.pid, signal.SIGUSR1)
        # video_writter.join()
        # df = pd.DataFrame(records, columns=['Time', 'Source', 'ID', 'Path'])
        # df.to_csv('report.csv', sep='\t')
        break    

cv2.destroyAllWindows()

