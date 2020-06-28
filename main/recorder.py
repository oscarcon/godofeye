import cv2
import sys
import signal

def signal_handler(signum, frame):
    global state
    if signum == signal.SIGUSR1:
        state = 'stop'

cap = cv2.VideoCapture(sys.argv[1])
video_writter = cv2.VideoWriter(f'video{sys.argv[2]}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (800,600))
state = 'run'
signal.signal(signal.SIGUSR1, signal_handler)

while state == 'run':
    try:
        ret, frame = cap.read()
        if ret:
            if ret:
                frame = cv2.resize(frame, (800,600))
                video_writter.write(frame)
    except KeyboardInterrupt:
        break

video_writter.release()
cap.release()