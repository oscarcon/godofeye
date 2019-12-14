import cv2
import queue
import logging
import threading

class Camera(cv2.VideoCapture):
    MAX_STACK_SIZE = 1024
    def __init__(self, source='rtsp://admin:dvt@12345@192.168.0.100:554/Streaming/Channels/101'):
        super().__init__(source)
        self.image_queue = queue.Queue(maxsize=self.MAX_STACK_SIZE)
        self.frame_width = self.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.video_writer = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (self.frame_width,self.frame_height))
        self.get_frame_thread = threading.Thread(target=self._get_frame)
        self.get_frame_thread.start()
        
    def _get_frame(self):
        try:
            while True:
                ret, frame = super().read()
                with self.image_queue.mutex:
                    self.image_queue.queue.clear()
                self.image_queue.put_nowait(frame)
        except:
            pass    # end thread
    def __del__(self):
        self.get_frame_thread.raise_exception()
        self.get_frame_thread.join()
    def read(self):
        try:
            return (True, self.image_queue.get())
        except:
            logging.debug("Queue empty")   
            return (False, [])