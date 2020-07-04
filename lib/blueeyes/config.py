import os

user_name = os.getlogin()

if user_name == 'huy':
    HOME = os.environ['HOME']
    EVIDENT_ROOT = '/home/huy/capstone/godofeye/etc/evidence'
    FRAME_COUNT_TO_DECIDE = 10
    MQTT_BROKER = 'localhost'
elif user_name == 'blueeyes1':
    HOME = os.environ['HOME']
    EVIDENT_ROOT = '/home/blueeyes1/test_options/static'
    FRAME_COUNT_TO_DECIDE = 10
    MQTT_BROKER = 'localhost'
