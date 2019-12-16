import cv2
import queue
import logging
import threading
from threading import Thread, Lock

class Camera(Thread):
    MAX_STACK_SIZE = 10
    lock = Lock()
    def __init__(self, source='rtsp://admin:dvt@12345@192.168.0.100:554/Streaming/Channels/101',frameskip=0):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(source)
        self.image_queue = queue.Queue(maxsize=self.MAX_STACK_SIZE)
        self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frameskip = frameskip
        # self.video_writer = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.frame_width,self.frame_height))
        # self.get_frame_thread = threading.Thread(target=self._get_frame)
        # self.get_frame_thread.start()
    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if ret:
                    self.image_queue.put(frame)
                # with self.image_queue.mutex:
                #     self.image_queue.queue.clear()
                # self.image_queue.put_nowait(frame)
        except Exception as e:
            logging.debug(e)
    # end thread
    # def _get_frame(self):
    #     try:
    #         while True:
    #             ret, frame = self.cap.read()
    #             if ret:
    #                 self.image_queue.put_nowait(frame)
    #             # with self.image_queue.mutex:
    #             #     self.image_queue.queue.clear()
    #             # self.image_queue.put_nowait(frame)
    #     except:
    #         pass    # end thread
    # def start_record(self):
    #     self.start()
    def set(self, parameter, value):
        if self.lock.acquire(False):
            self.cap.set(parameter, value)
            self.lock.release()
        if parameter == 'skip':
            self.frameskip = value
    def read(self):
        try:
            for _ in range(self.frameskip - 1):
                self.image_queue.get()
            frame = self.image_queue.get()
            self.image_queue.task_done()
            return (True, frame)
        except Exception as e:
            logging.debug(e)   
            return (False, [])
    def __del__(self):
        self.get_frame_thread.raise_exception()
        self.get_frame_thread.join()