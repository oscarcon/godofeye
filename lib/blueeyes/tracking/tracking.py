'''
Input: face location and encoded face feature buffer
Output: face location and id of face in buffer
'''
import os
import sys
import cv2
import queue
import time
import threading
import numpy as np

class Object:
    def __init__(self, box, feature):
        self.time = time.time()
        self.bounding_boxes = [box]
        self.features = [feature]
    def insert_box_feature(self, box, feature):
        self.bounding_boxes.append(box)
        self.features.append(feature)
    def get_location(self):
        box = self.bounding_boxes[-1]
        x = (box[1]+box[3])/2
        return x
    def live_time(self):
        return time.time() - self.time
    
class Tracking:
    def __init__(self, method, deadline = 400, threshold = 0.5, max_live_time = 10, **kwargs):
        self.buffer = []
        self.deadline = deadline
        self.threshold = threshold
        self.max_live_time = max_live_time
        self.method = method
        self.FRAME_WIDTH = kwargs['FRAME_WIDTH']
    def push(self, faces_info):
        if not self.buffer:
            for info in faces_info:
                box, feature = info
                self.buffer.append(Object(box, feature))
        else:
            for info in faces_info:
                self._asign_to_obj(info)
        # self._check_buffer_status()

    def count(self):
        return len(self.buffer)
    
    def features_history(self, i):
        if i < self.count():
            return self.buffer[i].features
        else:
            return None
    def box_history(self, i):
        if i < self.count():
            return self.buffer[i].bounding_boxes
        else:
            return None
    def clear(self):
        self.buffer = []

    def register_callback(self, cb_method):
        self.cb = cb_method

    def _self_check_buffer(self):
        self._check_buffer_status()
        threading.Timer(1.0, self._self_check_buffer).start()

    def _check_buffer_status(self):
        i = 0
        while i < len(self.buffer):
            if self.buffer[i].get_location() > self.deadline \
                    or self.buffer[i].live_time() > self.max_live_time \
                    or self.box_history(i)[-1][0] > self.FRAME_WIDTH*0.8 \
                    or self.box_history(i)[-1][0] < self.FRAME_WIDTH*0.2:
                del(self.buffer[i])
                i -= 1
            i += 1

    def _asign_to_obj(self, face_info):
        if self.method == 'feature':
            last_features = np.array([obj.features[-1] for obj in self.buffer])
            box, feature = face_info
            prepare_list = np.repeat([feature], len(last_features), axis=0)
            distances = np.linalg.norm(prepare_list-last_features, axis=1)
            if np.min(distances) < self.threshold:
                self.buffer[np.argmin(distances)].insert_box_feature(box, feature)
            else:
                self.buffer.append(Object(box, feature))
        if self.method == 'distance':
            centers = []
            for obj in self.buffer:
                x1, y1, x2, y2 = obj.bounding_boxes[-1]
                center = ( (x1+x2)/2, (y1+y2)/2 )
                centers.append(center) 
            box, feature = face_info
            x1, y1, x2, y2 = box
            stranger_center = ((x1+x2)/2, (y1+y2)/2)
            distances = [np.linalg.norm(np.array(stranger_center)-np.array(c)) for c in centers]
            i = np.argmin(distances)
            print(distances[i])
            if distances[i] < self.threshold:
                self.buffer[i].insert_box_feature(box, feature)
            else:
                self.buffer.append(Object(box, feature))
        if self.method == 'hybrid':
            pass

    