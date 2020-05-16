# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# Face detection using the YOLOv3 algorithm
#
# Description : yolo.py
# Contains methods of YOLO
#
# *******************************************************************

import os
import colorsys
import numpy as np
import cv2

from yolo.model import eval

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

class YOLO(object):
    ROOT_DIR = os.path.realpath(os.path.join(__file__, '../..'))
    # def __init__(self, args):
    #     self.args = args
    #         self.model_path = args.model
    #         self.classes_path = args.classes
    #         self.anchors_path = args.anchors
    #         self.class_names = self._get_class()
    #         self.anchors = self._get_anchors()
    #         self.sess = K.get_session()
    #         self.boxes, self.scores, self.classes = self._generate()
    #         self.model_image_size = args.img_size
    def __init__(self, model=ROOT_DIR+'/model-weights/YOLO_Face.h5',
        anchors=ROOT_DIR+'/cfg/yolo_anchors.txt',
        classes=ROOT_DIR+'/cfg/face_classes.txt',
        score=0.5, iou=0.45, img_size=(416,416)
    ):

        self.model_path = model
        self.classes_path = classes
        self.anchors_path = anchors
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.iou_threshold = iou
        self.score_threshold = score
        self.boxes, self.scores, self.classes = self._generate()
        self.model_image_size = img_size
       
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file'

        # load model, or construct model and load weights
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (
                           num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print(
            '*** {} model, anchors, and classes loaded.'.format(model_path))

        # generate colors for drawing bounding boxes
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # shuffle colors to decorrelate adjacent classes.
        np.random.seed(102)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names),
                                           self.input_image_shape,
                                           score_threshold=self.score_threshold,
                                           iou_threshold=self.iou_threshold)
        return boxes, scores, classes

    def detect(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[
                       0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[
                       1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(
                reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # add batch dimension
        image_data = np.expand_dims(image_data, 0)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
        result = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            result.append(tuple([int(elem) for elem in box]))
        return result

    def close_session(self):    
        self.sess.close()

def letterbox_image(image, size):
    '''Resize image with unchanged aspect ratio using padding'''
    # cv2.imshow('test', image)
    # cv2.waitKey(0)
    img_width, img_height = image.shape[1], image.shape[0]
    w, h = size
    scale = min(w / img_width, h / img_height)
    nw = int(img_width * scale)
    nh = int(img_height * scale)

    image = cv2.resize(image, (nw, nh))

    new_image = np.ones((h,w,3)) * 128
    y,x = ((h - nh) // 2), ((w - nw) // 2)
    new_image[y:y+nh, x:x+nw, :] = image
    new_image = new_image.astype('uint8')
    return new_image