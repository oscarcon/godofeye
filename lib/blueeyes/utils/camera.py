import cv2
import queue
import logging
import threading
from threading import Thread, Lock

logging.basicConfig(level=logging.DEBUG)

class Camera(Thread):
    lock = Lock()
    def __init__(self, source, frameskip=0):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(source)
        self.source = source
        self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frameskip = frameskip
        self.img_queue = queue.Queue(maxsize=1)
        self.state = 'run'
        # self.video_writer = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.frame_width,self.frame_height))
        # self.get_frame_thread = threading.Thread(target=self._get_frame)
        # self.get_frame_thread.start()
    def restart(self):
        self.cap = cv2.VideoCapture(self.source)
        
    def get(self, param):
        return self.cap.get(param)

    def isOpened(self):
        return self.cap.isOpened()
    def run(self):
        ret, frame = None, None
        frame_count = 0
        while self.state == 'run':
            ret, frame = self.cap.read()
            if ret:
                if frame_count % self.frameskip == 0:
                    if frame_count > 1000:
                        frame_count = 0
                    try:
                        self.img_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                frame_count += 1

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
        