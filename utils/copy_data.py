import os
import glob
import shutil

TRAIN_DATA_DIR = '../train_data/all'
DATASET_DIR = '../train_data/dataset'

if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)

for entry in os.scandir(TRAIN_DATA_DIR):
    print(entry.path)
    try:
        file_to_cp = glob.glob(os.path.join(entry.path, 'center_*.*'))[0]
    except Exception as e:
        print(e)
    dst_path = os.path.join(DATASET_DIR, entry.name)
    filename = os.path.basename(file_to_cp)
    os.makedirs(dst_path)
    shutil.copy(file_to_cp, os.path.join(dst_path, filename))
