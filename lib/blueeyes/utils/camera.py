import os
import cv2
import queue
import logging
import threading
from threading import Thread, Lock

logging.basicConfig(level=logging.DEBUG)

class Camera(Thread):
    lock = Lock()
    def __init__(self, source, frameskip=1, crop=(0.1, 0, 0, 0)):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(source)
        self.source = source
        self.source_frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
        self.source_frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = int(self.source_frame_width*(1 - crop[1]+crop[3]))
        self.frame_height = int(self.source_frame_height*(1 - crop[0]+crop[2]))
        self.frameskip = frameskip
        self.crop = crop
        self.img_queue = queue.Queue(maxsize=1)
        self.state = 'run'
        if os.path.exists(source):
            self.is_video = True
        else:
            self.is_video = False

    def restart(self):
        self.cap = cv2.VideoCapture(self.source)
        
    def get(self, param):
        if param == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frame_width
        if param == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frame_height
        return self.cap.get(param)

    def isOpened(self):
        return self.cap.isOpened()
    def run(self):
        frame_count = 0
        while self.state == 'run':
            ret, frame = self.cap.read()
            if ret:
                if frame_count % self.frameskip == 0:
                    if frame_count > 1000:
                        frame_count = 0
                    h1 = int(self.crop[0]*self.source_frame_height)
                    h2 = int((1-self.crop[2])*self.source_frame_height)
                    w1 = int(self.crop[3]*self.source_frame_width)
                    w2 = int((1-self.crop[1])*self.source_frame_width)
                    frame = frame[h1:h2, w1:w2]
                    try:
                        self.img_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                frame_count += 1
            elif self.is_video:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def set(self, parameter, value):
        if self.lock.acquire(False):
            self.cap.set(parameter, value)
            self.lock.release()
        if parameter == 'skip':
            self.frameskip = value
    def read(self):
        try:
            frame = self.img_queue.get()
            self.img_queue.task_done()    
            return (True, frame)
        except queue.Empty:
            print('empty image queue')
            return (False, [])
    def stop(self):
        self.state = 'stop'
    def __del__(self):
        self.stop()
        