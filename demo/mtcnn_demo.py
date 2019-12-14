import cv2
import time
from mtcnn import MTCNN

detector = MTCNN()

video = cv2.VideoCapture('/home/huy/Downloads/dataV10_oanh.avi')

while True:
    ret, frame = video.read()
    if ret:
        height, width, color_channel = frame.shape
        width_range = slice(int(0.47*width)-width//8, int(0.47*width)+width//8)
        height_range = slice(int(0.2*height),int(0.7*height))
        frame = frame[height_range, width_range, :]

        frame = cv2.resize(frame, (0,0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)

        faces  = detector.detect_faces(frame)
        for face in faces:
            x,y,width,height = face['box']
            cv2.rectangle(frame, (x,y), (x+width,y+height), color=(0,0,255), thickness=2)
        cv2.imshow('test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        frame_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos + 1)
cv2.destroyAllWindows()