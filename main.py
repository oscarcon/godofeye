import os
import sys
import cv2
from time import time, sleep
from queue import Queue
from collections import namedtuple

sys.path.append('./lib')
sys.path.append('./lib/yoloface')

from blueeyes.face_recognition import FaceDetector, FaceRecognition
from blueeyes.utils import Camera

import requests

# cap = Camera(source='/home/huy/Downloads/AnhDuy.mp4', frameskip=5)
cap = Camera(source='/home/huy/Downloads/Huy_14h50_BLCon.mp4', frameskip=0)
# cap = Camera(source='rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101', frameskip=5)

# cap = Camera(source=0, frameskip=15)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)

cap.start()

# Configurations
FRAME_COUNT_TO_DECIDE = 5

HOME = os.environ['HOME']
# detector = FaceDetector('mtcnn_torch', scale=8, min_face_size=60)
detector = FaceDetector('yolo', scale=1, model_img_size=(256,256))
# detector = YOLO()
recog = FaceRecognition(
    model_dir=f'{HOME}/face_recog/models/simple_distance',
    vggface=False,
    use_knn=False
    # retrain=False
)

def process_id(result_id):
    t = int(time())
    info = {
        'timestamp': t,
        'id': result_id,
        'link': f'./etc/evidence/{1}'
    }
    print(result_id)

recog.on_final_decision(process_id)

execution_time = {}

if not cap.cap.isOpened():
    print('Error in opening camera')
def main():
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                # frame = cv2.resize(frame, (0,0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)
        
                start_time = time()
                boxes = detector.detect(frame, size_ranges=[(70,90,100,150)])
                execution_time['detection'] = time() - start_time
                start_time = time()
                                        
                detector.debug(frame)

                labels = recog.recog(frame, boxes, threshold=0.4)
                execution_time['recognition'] = time() - start_time

                print(execution_time)

                # Post-processing the raw recognition data
                recog.put_to_result_buffer(boxes,labels)

                # show detected faces and recognized labels
                if labels and boxes:
                    for index,label in enumerate(labels):
                        location = (boxes[index][0], boxes[index][1])
                        name = label[0].replace('\n', '')
                        cv2.putText(frame, name, location, cv2.FONT_HERSHEY_PLAIN, 1.2, color=(255,0,0), thickness=2)
                frame_to_show = cv2.resize(frame, (0,0), fx=1/4, fy=1/4, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("frame", frame_to_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            break    

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
