import os
import sys
import cv2
import numpy as np
from time import time, sleep
from queue import Queue
from collections import namedtuple

sys.path.append('../lib')
sys.path.append('../lib/yoloface')

import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
oldinit = tf.Session.__init__
def new_tfinit(session, target='', graph=None, config=None):
    print("Set config.gpu_options.allow_growth to True")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    oldinit(session, target, graph, config)
tf.Session.__init__ = new_tfinit

from blueeyes.utils import Camera
from blueeyes.face_recognition import FaceDetector, FaceRecognition
from blueeyes.tracking import Tracking

# mqtt for communication between modules
import paho.mqtt.client as mqtt

mqtt_client = mqtt.Client()
mqtt_client.connect('localhost', 1883, 60)

# cap = Camera(source='/home/huy/Downloads/AnhDuy.mp4', frameskip=5)
cap = Camera(source='/home/huy/Downloads/Tung_nguoi_di_vao1.mp4', frameskip=0)
# cap = Camera(source='rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101', frameskip=5)
# cap = Camera(source=f'/home/huy/Downloads/{sys.argv[1]}.mp4', frameskip=0)
# cap = Camera(source='rtsp://Admin:12345@10.42.0.235:554/Streaming/Channels/101', frameskip=5)
# cap = Camera(source=0, frameskip=15)

# cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)

cap.start()

# Configurations
FRAME_COUNT_TO_DECIDE = 5

HOME = os.environ['HOME']
detector = FaceDetector('mtcnn', min_face_size=40)
# detector = FaceDetector('yolo', model_img_size=(128,128))
# detector = YOLO()
# recog = FaceRecognition(
#     model_dir=f'{HOME}/face_recog/models/simple_distance',
#     vggface=False,
#     use_knn=False
#     # retrain=False
# )

# recog = FaceRecognition(
#     model_dir='/home/huy/face_recog/models/knn',
#     classifier_method='knn'
# )
recog = FaceRecognition(
    classifier_method='svm',
    model_path='/home/huy/face_recog/models/svm/aug3_0.1_128.svm'
)

tracking = Tracking(deadline=cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.6, threshold=0.3, max_live_time=1000)

def process_id(result_id):
    mqtt_client.publish('/ids', result_id)
    print(result_id)

recog.on_final_decision(process_id)

execution_time = {}

if not cap.cap.isOpened():
    print('Error in opening camera')

frame_count = 0

def random_color():
    return tuple(np.random.choice(range(256), size=3))

color_table = {}

def percent_of_majority(lst):
    d = {}
    for ele in lst:
        if ele not in d:
            d[ele] = 0
        d[ele] += 1 
    keys, values = list(zip(*d.items()))
    max_count = max(values)
    return keys[values.index(max_count)], max_count/len(lst)

while True:
    try:
        mqtt_client.loop_start()
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (0,0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)
            frame_count += 1

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

            boxes = temp_boxes

            for box in boxes:
                face_img = recog.face_roi(frame, box)
                feature = recog.extract_feature(face_img)

                # Post-processing the raw recognition data
                # recog.put_to_result_buffer([box],[label])

                tracking.push([(box, feature)])
                # execution_time['recognition'] = time() - start_time
                # print(execution_time)
            if frame_count % 5 == 0:
                for i in range(tracking.count()):
                    features = tracking.features_history(i)
                    if len(features) > 10:
                        temp_labels = []
                        for feature in features:
                            label = recog.recog(feature, threshold=0.5)
                            temp_labels.append(label)
                        label, percent = percent_of_majority(temp_labels)
                        if percent > 0.7:
                            process_id(label)
                            
            # Show the frame to debug            
            for i in range(tracking.count()):
                if i not in color_table:
                    color = random_color()
                    color = tuple([int(x) for x in color])
                    color_table[i] = color
                detector.draw_bounding_box(frame, tracking.box_history(i), color_table[i])

            # show detected faces and recognized labels
            # if labels and boxes:
            #     for index,label in enumerate(labels):
            #         location = (boxes[index][0], boxes[index][1])
            #         name = label[0].replace('\n', '')
            #         cv2.putText(frame, name, location, cv2.FONT_HERSHEY_PLAIN, 1.2, color=(255,0,0), thickness=2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        break    

cv2.destroyAllWindows()

