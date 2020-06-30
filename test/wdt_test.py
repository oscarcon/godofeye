import os
import sys
from time import sleep
sys.path.append('../lib')

from blueeyes.utils import WDT

wdt = WDT(timeout=3)

def on_timeout():
    print('timeout')
    os._exit(-1)

wdt.on_timeout = on_timeout

while True:
    wdt.notify()
    sleep(5)
    print('exited')
