import os
import sys
import cv2
from time import time, sleep
from queue import Queue
from collections import namedtuple


sys.path.append('../lib')
sys.path.append('../lib/yoloface')
# sys.path.append('/home/huy/face_recog/yoloface')

# from yolo.yolo import YOLO

from blueeyes.face_recognition import FaceDetector, FaceRecognition
from blueeyes.utils import Camera
# cap = cv2.VideoCapture('rtsp://admin:be123456@192.168.1.15:554/Streaming/Channels/101')
# cap = Camera(source='/home/huy/Downloads/dataV13_group.avi', frameskip=0)
# cap = Camera(source='/home/huy/Downloads/AnhDuy.mp4', frameskip=5)
cap = Camera(source='/home/blueeyes1/Downloads/9h50.mp4', frameskip=0)
# cap = Camera(source='rtsp://admin:be123456@192.168.1.15:554/Streaming/Channels/101', frameskip=5)
# cap = Camera(source=0, frameskip=15)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
cap.start()
# cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)

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
# recog = Recognition(model_dir='/home/huy/code/godofeye/models', retrain=False)

# Point = namedtuple('Point', 'x y')

# background = {
#     'initial': False,
#     'roi': (),
#     'pixels': None
# }

# def static_vars(**kwargs):
#     def decorate(func):
#         for k in kwargs:
#             setattr(func, k, kwargs[k])
#         return func
#     return decorate

# @static_vars(lbmouse=0, p1=Point(0,0), p2=Point(0,0))
# def onMouse(event, x, y, flags, params):
#     # print(params)qq
#     if event == cv2.EVENT_LBUTTONDOWN:
#         onMouse.lbmouse = 1
#         onMouse.p1 = Point(x,y)
#     if event == cv2.EVENT_MOUSEMOVE:
#         if onMouse.lbmouse == 1:
#             onMouse.p2 = Point(x,y)
#     #         show = params[0].copy()
#     #         cv2.rectangle(show, onMouse.p1, (x,y), (255,0,0), 2)
#     #         cv2.imshow('Choose ROI', show)
#     #         cv2.waitKey(1)
#     if event == cv2.EVENT_LBUTTONUP:
#         onMouse.lbmouse = 0
#         onMouse.p2 = Point(x,y)

def face_tracking(box_buffer):
    pass

def detect_tuesday(box_buffer):
    pass

execution_time = {}


# result_buffer = [] # maxsize = 5

def process_id(result_id):
    print(result_id)

recog.on_final_decision(process_id)

while True:
    try:
        ret, frame = cap.read()
        if ret:
            # rescale_size = (int(frame.shape[1]*0.25), int(frame.shape[0]*0.25))
            # frame = cv2.resize(frame, rescale_size)
            frame = cv2.resize(frame, (0,0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)
            # height, width, color_channel = frame.shape
            # width_range = slice(int(0.47*width)-width//4, int(0.47*width)+width//4)
            # height_range = slice(int(0.2*height),int(0.6*height))
            # frame = frame[height_range, width_range, :]
            
            
            # Choosing ROI of background
            # while background['initial'] == False:
            #     cv2.namedWindow('Choose ROI')
            #     cv2.setMouseCallback('Choose ROI', onMouse)
            #     show = frame.copy()
            #     cv2.rectangle(show, onMouse.p1, onMouse.p2, (255,0,0), 2)
            #     cv2.imshow('Choose ROI', show)
            #     if  cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # background['initial'] = True
            # cv2.destroyWindow('Choose ROI')
            
            # background['roi'] = frame[onMouse.p1.y:onMouse.p2.y, onMouse.p1.x:onMouse.p2.x, :]
            # cv2.imshow('background', background['roi'])
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
                if brightness < 200:
                    temp_boxes.append(box)
            boxes = temp_boxes
                    
            detector.debug(frame)
            labels = recog.recog(frame, boxes, threshold=0.4)
            execution_time['recognition'] = time() - start_time

            print(execution_time)

            # Post-processing the raw recognition data
            recog.put_to_result_buffer(boxes,labels)
            '''
            num_person = len(boxes)
            num_result = 0
            # Add raw data to buffer
            if len(result_buffer) >= FRAME_COUNT_TO_DECIDE:
                result_buffer.pop(0)
            if num_person == 0:
                # result_buffer.append([])
                pass
            else: # more than 1 person
                result_buffer.append(labels)
            
            # if num_person == 2:
            #     print(labels)
            #     while True:
            #         pass

            # Check result buffer to decide what to print
            id_count = {}
            # Wait for the buffer to fill up and loop through buffer
            if len(result_buffer) >= FRAME_COUNT_TO_DECIDE:
                deep = max([len(lst) for lst in result_buffer])
                for row in range(deep):
                    for col in range(FRAME_COUNT_TO_DECIDE):
                        try:
                            ID = result_buffer[col][row][0]
                            if ID in id_count.keys():
                                id_count[ID] += 1
                            else:
                                id_count[ID] = 1
                        except IndexError:
                            break
                        else:
                            num_result += 1

            num_id = len(id_count.keys())
            if num_id < num_result:
                num_result = num_id
            # print(result_buffer)
            # print(num_result)
            # print(id_count)

            result_id = []
            if id_count:
                frequency_ids = [(k,v) for k, v in sorted(id_count.items(), key=lambda item: item[1])]
                for i in range(num_result):
                    if frequency_ids[i][1] > int(0.6*FRAME_COUNT_TO_DECIDE):
                        result_id.append(frequency_ids[i][0].replace('\n', ''))
            if result_id:
                print(result_id)
                with open('./id.txt', 'w') as f:
                    f.write('\n'.join(result_id))
            '''



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

