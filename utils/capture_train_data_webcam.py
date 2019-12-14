import os
import sys
import cv2
import glob
import numpy as np
from detector import Camera

SAVE_LOCATION = './train_data'

cam = Camera(0)

recording = False
count = 1

while True:
    ret, frame = cam.read()
    frame_to_show = frame.copy()
    if recording:
        cv2.putText(frame_to_show, 'Recording', (20,10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255,0,0), thickness=1)
    cv2.imshow("Data Capture", frame_to_show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recording = True
        name = input('Input name: ')
        save_dir = os.path.join(SAVE_LOCATION, name)
        os.makedirs(save_dir)
        count = 1
    if recording and key == ord('c'):
        cv2.imwrite(os.path.join(save_dir, f'{count}.jpg'), frame)
        cv2.imshow("Data Capture", 255*np.ones((frame.shape)))
        cv2.waitKey(10)
        count += 1
    if key == ord('q'):
        break

cam.release()