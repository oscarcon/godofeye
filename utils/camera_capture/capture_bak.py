#!/usr/bin/python3
import os
import cv2
import sys
import queue
import argparse
import threading
import numpy as np

shared_queue = queue.Queue()

parser = argparse.ArgumentParser('Dataset Builder Program')
parser.add_argument('filename', help='Filename for the output video.')
parser.add_argument('--user', help='Username for IP camera.')
parser.add_argument('--password', help='Password for IP camera.')
parser.add_argument('--ip', help='IP of the camera.')
parser.add_argument('--builtin-cam', action='store_true', help='Use internal camera.')
args = parser.parse_args()

if os.path.exists(args.filename):
    print(f'{args.filename} exists. Choose a different filename or rename')
    sys.exit(0)

if args.builtin_cam:
    print('0')
    cam_url = 0
else:
    cam_url = f'rtsp://{args.user}:{args.password}@{args.ip}:554/Streaming/Channels/101'
cap = cv2.VideoCapture(cam_url)

FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
FPS = 25
print(FRAME_HEIGHT,FRAME_WIDTH,FPS)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(args.filename, cv2.VideoWriter_fourcc(*'MJPG'), FPS, (FRAME_WIDTH, FRAME_HEIGHT))

# reading and showing image from camera thread
def get_frame(q):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            q.put(frame)
            # write the flipped frame
            show = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('frame', show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                while cv2.waitKey(1) & 0xFF != ord(' '):
                    pass
            if key == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
# save frame to video
def write_video(q):
    global out, cap
    while cap.isOpened() or not q.empty():
        out.write(q.get())
    out.release()
if __name__ == '__main__':
    thread1 = threading.Thread(target=get_frame, args=(shared_queue,))
    thread2 = threading.Thread(target=write_video, args=(shared_queue,))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()