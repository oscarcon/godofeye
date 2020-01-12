import os
import sys
import cv2
from time import time, sleep
from queue import Queue
from collections import namedtuple

sys.path.append('../lib')
sys.path.append('../lib/yoloface')

from blueeyes.face_recognition import FaceDetector, FaceRecognition
from blueeyes.utils import Camera

# cap = Camera(source='/home/huy/Downloads/AnhDuy.mp4', frameskip=5)
cap = Camera(source='/home/blueeyes1/Downloads/9h50.mp4', frameskip=0)
# cap = Camera(source='rtsp://admin:be123456@192.168.1.15:554/Streaming/Channels/101', frameskip=5)

# cap = Camera(source=0, frameskip=15)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)

cap.start()


# Configurations
FRAME_COUNT_TO_DECIDE = 5

HOME = os.environ['HOME']
detector = FaceDetector('mtcnn', min_face_size=60)
# detector = FaceDetector('yolo', model_img_size=(320,320))
# detector = YOLO()
recog = FaceRecognition(
    # model_dir='/home/huy/code/godofeye/models', 
    model_dir=f'{HOME}/Downloads',
    dataset=f'{HOME}/output',
    vggface=False,
    use_knn=True
    # retrain=False
)

def process_id(result_id):
    print(result_id)

recog.on_final_decision(process_id)

execution_time = {}

while True:
    try:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (0,0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)
     
            start_time = time()
            boxes = detector.detect(frame)
            execution_time['detection'] = time() - start_time
            start_time = time()
            
            # ignore too bright faces
            temp_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                hsv = cv2.cvtColor(frame[y1:y2,x1:x2,:], cv2.COLOR_BGR2HSV)
                brightness = cv2.mean(hsv)[2]
                if brightness < 100:
                    temp_boxes.append(box)
            boxes = temp_boxes
                    
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
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        break    

cv2.destroyAllWindows()

