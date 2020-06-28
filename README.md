# Face detection, recognition, tracking and post-processing system

## Folder structure
* lib/blueeyes: Contain all the modules of the system
    * face_detection: Face detection module source code
    * face_recogition: Face recognition module source code
    * tracking: Tracking module source code
* models: The location to store trained models
* utils: Scripts support the development and testing process
* main:
    * main.py: run this script to run the system, linked and used all the required modules: camera, recognition, etc. This script describes the logic of the program and error handling process.
    * train_model.ipynb (Jupyter Notebook): show the training processing step by step, from how to load dataset to trained model. Dataset, feature extraction algorithm, training algorithm and parameters can be configured here.

## Modules short description

## User guide
### Installation
1. Clone the repository.
2. Activate python virtual environment 
3. Install the requirement  
`pip install -r requirements.txt`

### How to run
1. Activate python virtual environment 
2. cd to ./main folder
3. Run $ python3 main.py

### How to train
Refer to main/train_model.ipynb