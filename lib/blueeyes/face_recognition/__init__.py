import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
from tensorflow.compat.v1.keras.backend import set_session

oldinit = tf.Session.__init__
def new_tfinit(session, target='', graph=None, config=None):
    print("Set config.gpu_options.allow_growth to True")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    oldinit(session, target, graph, config)
tf.Session.__init__ = new_tfinit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess) 

from .detector import FaceDetector
from .recognition import FaceRecognition, FeatureExtractor, ModelTraining