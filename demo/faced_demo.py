import cv2
from time import time
from faced import FaceDetector
from faced.utils import annotate_image
from detector import Camera

# cam = Camera(source=)
cam = cv2.VideoCapture('/home/huy/Downloads/dataV10_oanh.avi')

thresh = 0.9
face_detector = FaceDetector()
while True:
    ret, img = cam.read()
    try:
        if ret:
            # img = cv2.resize(img, (640,480))
            height, width, color_channel = img.shape
            width_range = slice(width//2-width//8, width//2+width//8)
            height_range = slice(int(0.2*height),int(0.8*height))
            img = img[height_range, width_range, :]
            rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

            # Receives RGB numpy image (HxWxC) and
            # returns (x_center, y_center, width, height, prob) tuples. 
            # for _ in range(3):
            #     start = time()
            #     print("Processing!!!")
            #     bboxes = face_detector.predict(rgb_img, thresh)
            #     print(time() - start)
            bboxes = face_detector.predict(rgb_img, thresh)
            # Use this utils function to annotate the image.
            ann_img = annotate_image(img, bboxes)
            # Show the image
            cv2.imshow('image',ann_img)
            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(e)
cv2.destroyAllWindows()