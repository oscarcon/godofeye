import os
import sys
from threading import Thread
from time import time, sleep
class WDT:
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.active = False
        self.thread = Thread(target=self._pooling)
        self.thread.start()
    def _pooling(self):
        while True:
            if self.active:
                if time() - self.t > self.timeout:
                    self.on_timeout()
                    break
            sleep(1)
    def notify(self):
        self.active = True
        self.t = time()
    def on_timeout(self):
        print('Reset')
        os._exit(-1)
    