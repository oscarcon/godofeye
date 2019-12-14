import os
import sys
import cv2
import shutil
import argparse
import numpy as np
from time import sleep, time

parser = argparse.ArgumentParser('''Image cropping program
For detail usage guide. Read the doc.
''')
parser.add_argument('name')
parser.add_argument('--folder', help='Provide image folder for getting image to crop')
parser.add_argument('--video', help='Video to capture frame')
parser.add_argument('--fps', type=int, help='Frame capturing rate')
parser.add_argument('--output', help='Output folder')

# START_DIR = int(sys.argv[1])
# END_DIR = int(sys.argv[2])

cmd_args = parser.parse_args()

class Arg:
    def __init__(self, args):
        self.fps = 10
        self.video = f'/home/huy/Downloads/dataV10_{args.name}.avi'
        self.output = f'/home/huy/output/{args.name}'

args = Arg(cmd_args)
delay = 1 / args.fps
print(delay)

lbmouse = 0
p1 = (0,0)
p2 = (0,0)

def onMouse(event, x, y, flags, params):
    global lbmouse, p1, p2
    if event == cv2.EVENT_LBUTTONDOWN:
        lbmouse = 1
        p1 = (x,y)
    if event == cv2.EVENT_MOUSEMOVE:
        if lbmouse == 1:
            show = clone.copy()
            cv2.rectangle(show, p1, (x,y), (255,0,0), 2)
            cv2.imshow('Cropper', show)
            cv2.waitKey(1)
    if event == cv2.EVENT_LBUTTONUP:
        lbmouse = 0
        p2 = (x,y)

def crop_image(img):
    height, width, color_channel = img.shape
    width_range = slice(width//2-width//8, width//2+width//8)
    height_range = slice(int(0.2*height),int(0.8*height))
    return img[height_range, width_range, :]
   
if args.video:
    # initialize variables
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    count = 0
    start_time = 0
    fx, fy = 1/3, 1/3   # define resize scale factor
    while cap.isOpened():
        if time() - start_time > delay:
            start_time = time()
            ret, frame = cap.read()
        elif ret==True:
            # write the flipped frame
            # clone = cv2.resize(frame, (0,0), fx=fx, fy=fy)
            clone = frame.copy()
            clone = crop_image(clone)
            cv2.imshow('Cropper',clone)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                cv2.namedWindow('Cropper')
                cv2.setMouseCallback('Cropper', onMouse, param=clone)
                while True:
                    cv2.imshow('Cropper', clone)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        height, width, color_channel = frame.shape
                        h_offset = int(0.2*height)
                        w_offset = width//2-width//8
                        cropped = frame[h_offset+p1[1]:h_offset+p2[1], w_offset+p1[0]:w_offset+p2[0], :]
                        try:
                            os.makedirs(os.path.join(args.output))
                        except:
                            pass
                        file_name = os.path.basename(args.video).split('.')[0]
                        cv2.imwrite(os.path.join(args.output, file_name + str(count) + '.jpg'), cropped)
                        count += 1
                        break
                    elif key == ord(' '):
                        break
            if key == ord('q'):
                break
        else:
            break
        

# for folder_path in [os.path.join('data', str(folder)) for folder in range(START_DIR, END_DIR+1)]:
#     folder_name = os.path.basename(folder_path)
#     for img_path in os.listdir(folder_path):
#         img = cv2.imread(os.path.join(folder_path, img_path))
#         clone = img.copy()
#         cv2.namedWindow('Cropper')
#         cv2.setMouseCallback('Cropper', onMouse)
#         cv2.imshow('Cropper', img)
#         key = cv2.waitKey(0) & 0xFF
#         if key == ord('c'):
#             cropped = clone[p1[1]:p2[1], p1[0]:p2[0], :]
#             try:
#                 os.makedirs(os.path.join('output', folder_name))
#             except:
#                 pass
#             cv2.imwrite(os.path.join('output', folder_name, img_path), cropped)
#             continue
#         if key == ord('q'):
#             cv2.destroyAllWindows()
#             exit()