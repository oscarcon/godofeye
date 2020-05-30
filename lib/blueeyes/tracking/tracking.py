'''
Input: face location and encoded face feature buffer
Output: face location and id of face in buffer
'''

import os
import sys
import cv2
import queue
import time

class Object:
    def __init__(self):
        self.time = time.time()
        self.bounding_boxes = []
        self.features = []
    def get_location(self):
        box = self.bounding_boxes[-1]
        x = (box[1]+box[3])/2
        return x
    def live_time(self):
        return time.time() - self.time()
    
class Tracking:
    def __init__(self):
        self.buffer = queue.Queue()
        self.n_group = 0
        self.dead_line = 400
        pass
    def push(self, faces_info):

        pass
    def clear(self):
        self.buffer = queue.Queue()
    def register_callback(self, cb_method):
        self.cb = cb_method
    
    