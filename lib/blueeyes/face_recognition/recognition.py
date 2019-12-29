import os
import cv2
import sys
import pdb
import enum
import pickle
# import msgpack
import numpy as np
import keras_vggface
import face_recognition
from scipy.spatial import distance
from keras_vggface.utils import preprocess_input

class TrainOption(enum.Enum):
    RETRAIN = 1
    UPDATE = 2
    RUNONLY = 3

class FaceRecognition:
    def __init__(self, model_dir='', vggface=False ,dataset='/home/huy/code/godofeye/train_data/dataset_be_ok', trainopt=TrainOption.RUNONLY):
        self.dataset = dataset
        self.model_dir = model_dir
        self.vggface = vggface
        if vggface:
            self.vgg_model = keras_vggface.VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

        if not os.path.exists(os.path.join(model_dir, 'model.dat')) or trainopt==TrainOption.RETRAIN:
            self.model = []
            self.labels = []
            self._create_model()
        elif trainopt == TrainOption.UPDATE:
            self._create_model(TrainOption.UPDATE)
        elif trainopt == TrainOption.RUNONLY:
            self._load_model()
            # with open(model_dir, 'rb') as raw:
            #     model_label = pickle.load(raw)
            #     self.model = [[x] for x in model_label['features']]
            #     self.labels = [[x] for x in model_label['labels']]
    def _vgg_encoding(self, image):
        sample = cv2.resize(image, (224,224))
        sample = sample.astype('float32')
        sample = np.expand_dims(sample, axis=0)
        sample = preprocess_input(sample, version=2)
        yhat = self.vgg_model.predict(sample)
        return yhat
    def _create_model(self, trainopt=TrainOption.RETRAIN):
        # pdb.set_trace()
        sample_per_class = 5
        raw_labels_file = open(os.path.join(self.model_dir, 'labels.dat'), 'w')
        raw_model_file = open(os.path.join(self.model_dir, 'model.dat'), 'wb')
        for entry in os.scandir(self.dataset):
            if entry.is_dir(): 
                self.labels.append(entry.name)
                raw_labels_file.write(entry.name + '\n')
                # list for putting multiple encoded vector of a same person
                encoded_vec_list = []
                sample_count = -1
                for path in os.listdir(entry.path):
                    sample_count += 1
                    if sample_count == sample_per_class:
                        break
                    try:
                        img = face_recognition.load_image_file(os.path.join(entry.path,path))
                        if self.vggface:
                            encoded_vec = self._vgg_encoding(img)[0]
                        else:
                            bounding_box = face_recognition.face_locations(img)
                            if isinstance(img, np.ndarray):
                                known_face_box = [(0, img.shape[1], img.shape[0], 0)]
                            else:
                                raise TypeError
                            # encoded_vec = face_recognition.face_encodings(img, bounding_box)[0]
                            encoded_vec = face_recognition.face_encodings(img, known_face_locations=known_face_box)[0]
                        print(encoded_vec)
                        encoded_vec_list.append(encoded_vec)
                    except Exception as e:
                        print(e)
                encoded_vec_list = [np.average(encoded_vec_list, axis=0)]
                self.model.append(encoded_vec_list)
        np.save(raw_model_file, self.model)

    def _load_model(self):
        with open(os.path.join(self.model_dir, 'model.dat'), 'rb') as raw_model_file:
            self.model = np.load(raw_model_file, allow_pickle=True)
        with open(os.path.join(self.model_dir, 'labels.dat'), 'r') as raw_labels_file:
            self.labels = raw_labels_file.readlines()
    def _vgg_recog(self, frame, boxes, recog_level=1, threshold=0.5):
        result = []
        for (x1, y1, x2, y2) in boxes:
            # print(x1,y1,x2,y2)
            crop = frame[y1:y2, x1:x2, :]
            cv2.imshow('debug', crop)
            cv2.waitKey(0)
            target_face = self._vgg_encoding(crop)[0]
            min_distance = 1000
            predict_label = ['unknown']
            for model_face_list, label in zip(self.model, self.labels):
                dis = distance.euclidean(target_face, model_face_list[0])
                if dis < min_distance:
                    min_distance = dis
                    predict_label = [label]
            if min_distance > threshold:
                result.append(['unknown'])
            else:
                result.append(predict_label)
        return result

    def _adam_recog(self, frame, boxes, recog_level=1, threshold=0.5):
        result = []
        for (x1, y1, x2, y2) in boxes:
            # print(x1,y1,x2,y2)
            # crop = frame[y1:y2, x1:x2, :]
            # cv2.imshow('debug', crop)
            # cv2.waitKey(0)
            target_face = face_recognition.face_encodings(frame, known_face_locations=[(y1,x2,y2,x1)])[0]
            match_count = 0
            total_count = 0
            predict_label = ['unknown']
            # slow method (check pass)
            # min_distance = 1
            # for model_face_list, label in zip(self.model, self.labels):
            #     dis = distance.euclidean(target_face, model_face_list[0])
            #     if dis < min_distance:
            #         min_distance = dis
            #         predict_label = [label]

            # experimental
            prepare_list = np.repeat([[target_face]], len(self.model), axis=0)
            result_list = np.linalg.norm(prepare_list-self.model, axis=2)
            min_distance = np.min(result_list)
            # print(min_distance, np.argmin(result_list))
            # print(prepare_list.shape, self.model.shape, result_list.shape)
            predict_label = [self.labels[np.argmin(result_list)]]
            
            if min_distance > threshold:
                result.append(['unknown'])
            else:
                result.append(predict_label)
        return result
        #     for model_face_list, label in zip(self.model, self.labels):
        #         if match_count > 0:
        #             break
        #         else:
        #             match_count = 0
        #             total_count = 0
        #         for i in range(recog_level):
        #             try:
        #                 match = face_recognition.compare_faces([model_face_list[i]], target_face, tolerance=threshold)
        #             except Exception as e:
        #                 print(e)
        #                 print(model_face_list[i])
        #                 print(target_face)
        #             else:
        #                 total_count += 1
        #                 if match == True:
        #                     match_count += 1
        #         if total_count != 0:
        #             if match_count/total_count > 0.5:
        #                 result.append([label])
        #     if not match_count:
        #         result.append(['unknown'])
        # return result
    def recog(self, frame, boxes, **kwargs):
        if self.vggface:
            result = self._vgg_recog(frame, boxes, **kwargs)
        else:
            result = self._adam_recog(frame, boxes, **kwargs)
        return result 
if __name__ == '__main__':
    recog = Recognition()
