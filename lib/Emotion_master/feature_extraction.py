import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from Emotion_master.utils.datasets import get_labels
from Emotion_master.utils.inference import detect_faces
from Emotion_master.utils.inference import draw_text
from Emotion_master.utils.inference import draw_bounding_box
from Emotion_master.utils.inference import apply_offsets
from Emotion_master.utils.inference import load_detection_model
from Emotion_master.utils.preprocessor import preprocess_input
USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = '/home/blueeyes1/godofeye/lib/Emotion_master/models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

def feature_extraction(gray_face,origin_frame):
    gray_image = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2GRAY)
    for face_coordinates in gray_face:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        print(gray_face)
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            import logging
            logging.warning("resize emotion_target")
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        #print(gray_face.shape)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_prediction = emotion_prediction.tolist()
        emotion_prediction[0][0:3]=[0,0,0]
        emotion_prediction[0][4:6]=[0,0]
        #emotion_prediction[0][0:3]=[0,0,0]
        #emotion_prediction[0][5]=0
        emotion_probability = np.max(emotion_prediction)
        
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        return emotion_text