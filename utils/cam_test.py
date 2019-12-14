import os
import cv2
import detector
# video = cv2.VideoCapture('../person2.avi')
# video.set(cv2.CAP_PROP_POS_FRAMES, 300)
cam = detector.Camera()
face_detector = cv2.CascadeClassifier('cascade_model/cascade_ignore_shirt.xml')



while True:
    ret, frame = cam.read()
    if ret == None:
        break
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    # for (x,y,w,h) in faces:
    #     print(w,h)
    #     frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow('demo', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)

cv2.destroyAllWindows()