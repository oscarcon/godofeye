import os
import sys
import cv2
from collections import namedtuple


sys.path.append('../lib')

from blueeyes.face_recognition import FaceDetector, FaceRecognition

# video = cv2.VideoCapture('test_video/person1.avi')
# video.set(cv2.CAP_PROP_FPS, 10)
# cap = Camera()
cap = cv2.VideoCapture('/home/huy/Downloads/dataV13_group.avi')
# cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
detector = FaceDetector('mtcnn', min_face_size=50)
recog = FaceRecognition(
    model_dir='/home/huy/code/godofeye/models', 
    dataset='/home/huy/output',
    vggface=False, 
    retrain=False
)
# recog = Recognition(model_dir='/home/huy/code/godofeye/models', retrain=False)

Point = namedtuple('Point', 'x y')

background = {
    'initial': False,
    'roi': (),
    'pixels': None
}

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(lbmouse=0, p1=Point(0,0), p2=Point(0,0))
def onMouse(event, x, y, flags, params):
    # print(params)qq
    if event == cv2.EVENT_LBUTTONDOWN:
        onMouse.lbmouse = 1
        onMouse.p1 = Point(x,y)
    if event == cv2.EVENT_MOUSEMOVE:
        if onMouse.lbmouse == 1:
            onMouse.p2 = Point(x,y)
    #         show = params[0].copy()
    #         cv2.rectangle(show, onMouse.p1, (x,y), (255,0,0), 2)
    #         cv2.imshow('Choose ROI', show)
    #         cv2.waitKey(1)
    if event == cv2.EVENT_LBUTTONUP:
        onMouse.lbmouse = 0
        onMouse.p2 = Point(x,y)

while True:
    try:
        ret, frame = cap.read()
        if ret:
            rescale_size = (int(frame.shape[1]*0.25), int(frame.shape[0]*0.25))
            # frame = cv2.resize(frame, rescale_size)
            height, width, color_channel = frame.shape
            width_range = slice(int(0.47*width)-width//8, int(0.47*width)+width//8)
            height_range = slice(int(0.2*height),int(0.6*height))
            frame = frame[height_range, width_range, :]
            # frame = cv2.resize(frame, (0,0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)

            while background['initial'] == False:
                cv2.namedWindow('Choose ROI')
                cv2.setMouseCallback('Choose ROI', onMouse)
                show = frame.copy()
                cv2.rectangle(show, onMouse.p1, onMouse.p2, (255,0,0), 2)
                cv2.imshow('Choose ROI', show)
                if  cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            background['initial'] = True
            cv2.destroyWindow('Choose ROI')
            
            background['roi'] = frame[onMouse.p1.y:onMouse.p2.y, onMouse.p1.x:onMouse.p2.x, :]
            cv2.imshow('background', background['roi'])

            boxes = detector.detect(frame)
            detector.debug(frame)
            labels = recog.recog(frame, boxes, threshold=0.5)
            if labels and boxes:
                for index,label in enumerate(labels):
                    location = (boxes[index][0], boxes[index][1])
                    name = label[0].replace('\n', '')
                    cv2.putText(frame, name, location, cv2.FONT_HERSHEY_PLAIN, 1.2, color=(255,0,0), thickness=2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        break    
cv2.destroyAllWindows()