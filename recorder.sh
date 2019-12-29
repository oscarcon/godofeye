#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/huy/code/godofeye/lib
python3 ./utils/camera_capture/capture.py $1.avi --ip 10.10.46.139 \
    --user admin --password be123456 \
    --output /home/huy/code/godofeye/utils/camera_capture/captured_images
