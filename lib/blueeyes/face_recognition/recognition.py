import os
import cv2
import sys
import pdb
import enum
import pickle
import traceback
import threading
import numpy as np
import keras_vggface
import face_recognition
from scipy.spatial import distance
from keras_vggface.utils import preprocess_input

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier

class TrainOption(enum.Enum):
    RETRAIN = 1
    UPDATE = 2
    RUNONLY = 3

class FaceRecognition:
    FRAME_COUNT_TO_DECIDE = 10
    def __init__(self, model_dir='', use_knn=False, knn_opts=(7, 'euclidean'), vggface=False ,dataset='/home/huy/code/godofeye/train_data/dataset_be_ok', trainopt=TrainOption.RUNONLY):
        self.dataset = dataset
        self.model_dir = model_dir
        self.vggface = vggface
        self.use_knn = use_knn
        self.result_buffer = []
        if vggface:
            self.vgg_model = keras_vggface.VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
        if not os.path.exists(os.path.join(model_dir, 'model.dat')) or trainopt==TrainOption.RETRAIN:
            self.model = []
            self.labels = []
            self._create_model()
        elif trainopt == TrainOption.UPDATE:
            self._create_model(TrainOption.UPDATE)
        elif trainopt == TrainOption.RUNONLY:
            if self.use_knn == True:
                self._load_model(model_dir=model_dir)
            else:
                self._load_model()
            # with open(model_dir, 'rb') as raw:
            #     model_label = pickle.load(raw)
            #     self.model = [[x] for x in model_label['features']]
            #     self.labels = [[x] for x in model_label['labels']]
    def config_postprocessing(self, **kwargs):
        FRAME_COUNT_TO_DECIDE = kwargs['FRAME_COUNT_TO_DECIDE']

    @staticmethod
    def train_knn(train_set_dict, K=7, metric='euclidean', output_model_location='.'):
        knn = KNeighborsClassifier(n_neighbors=K, metric=metric)
        model_pkl = open(os.path.join(output_model_location, 'knn_clf.pkl'), 'wb')
        features = []
        labels = []
        for id, img_paths in train_set_dict.items():
        # pdb.set_trace()
            for img_path in img_paths:
                labels.append(id)
                try:
                    img = face_recognition.load_image_file(img_path)
                    # if vggface:
                    #     encoded_vec = self._vgg_encoding(img)[0]
                    if False: # reserver for vggface condition check
                        pass
                    else:
                        # bounding_box = face_recognition.face_locations(img)
                        # encoded_vec = face_recognition.face_encodings(img, bounding_box)[0]
                        if isinstance(img, np.ndarray):
                            known_face_box = [(0, img.shape[1], img.shape[0], 0)]
                        else:
                            raise TypeError
                        encoded_vec = face_recognition.face_encodings(img, known_face_locations=known_face_box)[0]
                    print(encoded_vec)
                    features.append(encoded_vec)
                except Exception as e:
                    print(e)
        knn.fit(features, labels)
        print(knn)
        pickle.dump(knn, model_pkl)

    @staticmethod
    def train_model(train_set_dict, output_model_location='.'):
        labels = []
        model = []
        raw_labels_file = open(os.path.join(output_model_location, 'labels.dat'), 'w')
        raw_model_file = open(os.path.join(output_model_location, 'model.dat'), 'wb')
        for id, img_paths in train_set_dict.items():
        # pdb.set_trace()
            labels.append(id)
            raw_labels_file.write(id + '\n')
            for img_path in img_paths:
                # list for putting multiple encoded vector of a same person
                encoded_vec_list = []
                try:
                    img = face_recognition.load_image_file(img_path)
                    # if vggface:
                    #     encoded_vec = self._vgg_encoding(img)[0]
                    if False: # reserver for vggface condition check
                        pass
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
            model.append(encoded_vec_list)
        np.save(raw_model_file, model)
        raw_labels_file.close()
        raw_model_file.close()
        
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

    def _load_model(self, **kwargs):
        if self.use_knn == True:
            self.knn = pickle.load(open(kwargs['model_dir'] + '/knn_clf.pkl', 'rb'))
        else:
            with open(os.path.join(self.model_dir, 'model.dat'), 'rb') as raw_model_file:
                self.model = np.load(raw_model_file, allow_pickle=True)
            with open(os.path.join(self.model_dir, 'labels.dat'), 'r') as raw_labels_file:
                self.labels = raw_labels_file.readlines()
    
    def _knn_recog(self, frame, boxes, **kwargs):
        result = []
        for (x1, y1, x2, y2) in boxes:
            try:
                # print(x1,y1,x2,y2)
                # crop = frame[y1:y2, x1:x2, :]
                # cv2.imshow('debug', crop)
                # cv2.waitKey(0)
                target_face = face_recognition.face_encodings(frame, known_face_locations=[(y1,x2,y2,x1)])[0]
                probas = self.knn.predict_proba([target_face])
                if np.max(probas) >= kwargs['threshold']:
                    label = self.knn.classes_[np.argmax(probas)]
                    result.append([label])
                else:
                    result.append(['unknown'])
            except:
                traceback.print_exc()
        return result
            
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
            try:
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
            except:
                print(result_list)
                traceback.print_exc()
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
        if self.use_knn:
            result = self._knn_recog(frame, boxes, **kwargs)
        elif self.vggface:
            result = self._vgg_recog(frame, boxes, **kwargs)
        else:
            result = self._adam_recog(frame, boxes, **kwargs)
        return result 

    def put_to_result_buffer(self, boxes, labels):
        num_person = len(boxes)
        # Add raw data to buffer
        if len(self.result_buffer) >= self.FRAME_COUNT_TO_DECIDE:
            self.result_buffer.pop(0)
        if num_person == 0:
            # result_buffer.append([])
            pass
        else: # more than 1 person
            self.result_buffer.append(labels)
        self._event.set()

    def postprocessing(self, event, callback):
        try:
            while True:
                # if num_person == 2:
                #     print(labels)
                #     while True:
                #         pass
                event.wait()
                num_result = 0
                # Check result buffer to decide what to print
                id_count = {}
                # Wait for the buffer to fill up and loop through buffer
                if len(self.result_buffer) >= self.FRAME_COUNT_TO_DECIDE:
                    deep = max([len(lst) for lst in self.result_buffer])
                    for row in range(deep):
                        for col in range(self.FRAME_COUNT_TO_DECIDE):
                            try:
                                ID = self.result_buffer[col][row][0]
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
                        if frequency_ids[i][1] > int(0.6*self.FRAME_COUNT_TO_DECIDE):
                            result_id.append(frequency_ids[i][0].replace('\n', ''))
                if len(result_id) > 0:
                    callback(result_id)
                event.clear()
        except:
            traceback.print_exc()

    def on_final_decision(self, callback):
        self._event = threading.Event()
        self._thread = threading.Thread(target=self.postprocessing, args=(self._event, callback))
        self._thread.start()

if __name__ == '__main__':
    recog = Recognition()
