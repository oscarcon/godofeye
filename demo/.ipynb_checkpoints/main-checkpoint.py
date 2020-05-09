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
# cap = Camera(source='/home/huy/Downloads/3_nguoi_trucdien_DatHuyNhat.mp4', frameskip=0)
# cap = Camera(source='rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101', frameskip=5)

# cap = Camera(source=f'/home/huy/Downloads/{sys.argv[1]}.mp4', frameskip=0)
cap = Camera(source='rtsp://Admin:12345@10.42.0.235:554/Streaming/Channels/101', frameskip=5)


# cap = Camera(source=0, frameskip=15)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)

cap.start()

# Configurations
FRAME_COUNT_TO_DECIDE = 5

HOME = os.environ['HOME']
detector = FaceDetector('mtcnn', min_face_size=60)
# detector = FaceDetector('yolo', model_img_size=(128,128))
# detector = YOLO()
# recog = FaceRecognition(
#     model_dir=f'{HOME}/face_recog/models/simple_distance',
#     vggface=False,
#     use_knn=False
#     # retrain=False
# )

recog = FaceRecognition(
    classifier_method='nn'
)

def process_id(result_id):
    print(result_id)

recog.on_final_decision(process_id)

execution_time = {}

if not cap.cap.isOpened():
    print('Error in opening camera')

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
                if 75 < brightness < 225:
                    temp_boxes.append(box)
            boxes = temp_boxes
                    
            detector.debug(frame)
            labels = recog.recog(frame, boxes, threshold=0.0)
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

