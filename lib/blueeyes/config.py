import os

user_name = os.getlogin()

SHOW_FRAME = True

if user_name == 'huy':
    HOME = os.environ['HOME']
    EVIDENT_ROOT = '/home/huy/capstone/godofeye/etc/evidence'
    FRAME_COUNT_TO_DECIDE = 10
    MQTT_BROKER = 'localhost'
    SVM_MODEL_PATH = '/home/huy/face_recog/models/svm/rbf_c100_12062020_214524.svm'
elif user_name == 'blueeyes1':
    HOME = os.environ['HOME']
    EVIDENT_ROOT = '/home/blueeyes1/test_options/static'
    FRAME_COUNT_TO_DECIDE = 10
    MQTT_BROKER = 'localhost'
    SVM_MODEL_PATH = '/home/blueeyes1/rbf_c100_12062020_214524.svm'
