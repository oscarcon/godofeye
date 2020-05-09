import tensorflow as tf
oldinit = tf.Session.__init__
def new_tfinit(session, target='', graph=None, config=None):
    print("Set config.gpu_options.allow_growth to True")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    oldinit(session, target, graph, config)
tf.Session.__init__ = new_tfinit
import face_recognition
import cv2
from PIL import Image
import numpy as np
import urllib3
from io import BytesIO
import sys
import argparse
import cv2
import time
import pygame
import os
import ast
#from goto import with_goto
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta
import face_recognition
from faces import FaceDetector
from data import FaceData
from gabor import GaborBank
from emotions import EmotionsDetector

from http.server import BaseHTTPRequestHandler, HTTPServer

import threading
import queue
import random
from flask import Flask , jsonify,render_template
import pymysql
import base64
# import keyboard as kb
import traceback

sys.path.append('../godofeye/lib')
sys.path.append('../godofeye/lib/yoloface')

from blueeyes.face_recognition import FaceDetector, FaceRecognition
from blueeyes.utils import Camera

cap = Camera(source='rtsp://admin:be123456@10.10.46.224:554/Streaming/Channels/101', frameskip=5)
cap.start()

# Configurations
FRAME_COUNT_TO_DECIDE = 5

HOME = os.environ['HOME']
# detector = FaceDetector('mtcnn', min_face_size=60)
detector = FaceDetector('yolo', model_img_size=(128,128))
# detector = YOLO()
recog = FaceRecognition(
    # model_dir='/home/huy/code/godofeye/models', 
    model_dir=f'{HOME}/Downloads',
    dataset=f'{HOME}/output',
    vggface=False,
    use_knn=True
    # retrain=False
)

def process_id(result_id):
    print(result_id)

recog.on_final_decision(process_id)

execution_time = {}

app = Flask(__name__)

# video = cv2.VideoCapture(0)
#frame_width = int(video.get(3))
#frame_height = int(video.get(4))
# FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# FPS = int(cap.get(cv2.CAP_PROP_FPS))
# FPS = 25
# cap = cv2.VideoWriter('result12.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS,(FRAME_WIDTH,FRAME_HEIGHT))

#relax_ init
def music(tenbaihat):
    pygame.mixer.init()
    pygame.mixer.music.load(tenbaihat)
    pygame.mixer.music.play()
#---------------------------------------------
def splitname(str_):
    str_1 = list(str_)
    result = str_1[0].upper()
    for i in range(1,len(str_1)):
        if str_1[i].isupper():

            result+=" "+str_1[i]
        else:
            result+=str_1[i]
    return result

class VideoData:

    """
    Helper class to present the detected face region, landmarks and emotions.
    """

    #-----------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._faceDet = FaceDetector()
        '''
        The instance of the face detector.
        '''

        self._bank = GaborBank()
        '''
        The instance of the bank of Gabor filters.
        '''

        self._emotionsDet = EmotionsDetector()
        '''
        The instance of the emotions detector.
        '''

        self._face = FaceData()
        '''
        Data of the last face detected.
        '''

        self._emotions = OrderedDict()
        '''
        Data of the last emotions detected.
        '''

    #-----------------------------------------
    def detect(self, frame):
        """
        Detects a face and the prototypic emotions on the given frame image.

        Parameters
        ----------
        frame: numpy.ndarray
            Image where to perform the detections from.

        Returns
        -------
        ret: bool
            Indication of success or failure.
        """

        ret, face = self._faceDet.detect(frame)
        if ret:
            self._face = face

            # Crop just the face region
            frame, face = face.crop(frame)

            # Filter it with the Gabor bank
            responses = self._bank.filter(frame)

            # Detect the prototypic emotions based on the filter responses
            self._emotions = self._emotionsDet.detect(face, responses)
            
            return self._emotions
        else:
            self._face = None
            return False

    #-----------------------------------------
    def draw(self, frame):
        """
        Draws the detected data of the given frame image.

        Parameters
        ----------
        frame: numpy.ndarray
            Image where to draw the information to.
        """
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        glow = 3 * thick

        # Color settings
        black = (0, 0, 0)
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        red = (0, 0, 255)

        empty = True

        # Plot the face landmarks and face distance
        x = 5
        y = 0
        w = int(frame.shape[1]* 0.2)

        try:
            face = self._face
            empty = face.isEmpty()
            #face.draw(frame)
        except:
            traceback.print_exc()
            pass

        # Plot the emotion probabilities
        try:
            emotions = self._emotions
            if empty:
                labels = []
                values = []
            else:
                labels = list(emotions.keys())
                values = list(emotions.values())
                bigger = labels[values.index(max(values))]

                # Draw the header
                text = 'emotions'
                size, _ = cv2.getTextSize(text, font, scale, thick)
                y += size[1] + 20

                cv2.putText(frame, text, (x, y), font, scale, black, glow)
                cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

                y += 5
                cv2.line(frame, (x,y), (x+w,y), black, 1)

            size, _ = cv2.getTextSize('happiness', font, scale, thick)
            t = size[0] + 20
            w = 150
            h = size[1]
            for l, v in zip(labels, values):
                lab = '{}:'.format(l)
                val = '{:.2f}'.format(v)
                size, _ = cv2.getTextSize(l, font, scale, thick)

                # Set a red color for the emotion with bigger probability
                color = red if l == bigger else yellow

                y += size[1] + 15

                p1 = (x+t, y-size[1]-5)
                p2 = (x+t+w, y-size[1]+h+5)
                cv2.rectangle(frame, p1, p2, black, 1)

                # Draw the filled rectangle proportional to the probability
                p2 = (p1[0] + int((p2[0] - p1[0]) * v), p2[1])
                cv2.rectangle(frame, p1, p2, color, -1)
                cv2.rectangle(frame, p1, p2, black, 1)

                # Draw the emotion label
                cv2.putText(frame, lab, (x, y), font, scale, black, glow)
                cv2.putText(frame, lab, (x, y), font, scale, color, thick)

                # Draw the value of the emotion probability
                cv2.putText(frame, val, (x+t+5, y), font, scale, black, glow)
                cv2.putText(frame, val, (x+t+5, y), font, scale, white, thick)
        except Exception as e:
            print(e)
            pass


### The main function ###

def main(argv,dem_nhac=0):
    global video
    def distance_cal(f,h,p):
        return f*h/p
    button = 0
    def button_():
        nonlocal button
        button = int(input("Would you like input (1/0) :"))
    
    name = ""

    if dem_nhac >= 1:
        music("relax.mp3")
    bien=0

    Q = []
    x = "image_css"
    allfiles = os.listdir(x)
    dem_x = 0
    list_image_css = []
    list_names = []
    for i in allfiles :
        dem_x += 1
    for i in range(dem_x) :
        n=0
        try:
            allfiles_1 = os.listdir ( x + '/' + allfiles[i] )
        except:
            traceback.print_exc()
            continue
        for j in allfiles_1:
            if os.path.exists('variable/'+allfiles[i]+j+'.txt'):
                #print(n)
                fileface = open('variable/'+allfiles[i]+j+'.txt','r+')
                a=fileface.read()
                a= a.replace('  ', ' ')
                a = a.replace('  ',' ')
                a = a.replace('  ',' ')
                a = a.replace(' ',',')
                a = a+'x'
                try:
                    b = eval(a.split("x")[0])
                    list_image_css.append(b)
                    fileface.close()
                    n+=1
                except:
                    traceback.print_exc()
                    pass
                

            elif not os.path.exists('variable/'+allfiles[i]+j+'.txt'):
                a=face_recognition.load_image_file(x+'/'+allfiles[i]+'/'+j)
                try:
                    b=face_recognition.face_encodings(a)[0]#,known_face_locations=[[0,a.shape(1),a.shape(0),0]])[0]
                    fileface = open('variable/'+allfiles[i]+j+'.txt','w+')
                    fileface.write(str(b))
                    fileface.close()
                    list_image_css.append(b)
                    n+=1
                except:
                    traceback.print_exc()
                    pass

        for k in range(n):
            list_names.append(allfiles[i])
    #print(list_image_css)
    rutgon_ = []
    
    for i in list_names:
        if i not in rutgon_:
            rutgon_.append(i)
    read_rutgon_ = open('best/listsv.txt','w+')
    for i in rutgon_:
        read_rutgon_.write("\n"+i)
    read_rutgon_.close()

    read_chuyencan_ = open('best/chuyencan.txt','w+')
    for i in range(len(rutgon_)):
        read_chuyencan_.write("\n"+str(0))
    read_chuyencan_.close()

    # Create arrays of known face encodings and their names
    known_face_encodings = list_image_css
    known_face_names = list_names
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # Parse the command line
    # Create the helper class
    # data = VideoData() #// camxuc

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thick = 1
    glow = 3 * thick

    # Color settings
    color = (255, 255, 255)

    paused = False
    frameNum = 0
    k=[]
    # Process the video input
    bien_=0
    #list_names = []
    thuongxuyen_ = 0
    hiendien = 0
    hiendien_ten = []
    vang_ = len(list_names)
    file_vang = []
    bien_dem = 0
    list_buon = [""]
    list_vang_vuaroi = []
    
    list_name_camxuc = []
   
    unknow_count=0
    while True:
        localtime = time.asctime(time.localtime(time.time()))
        
        if not paused:
            start = datetime.now()  
        else:
            paused = True
        if ret:
            ret, frame = camera.read()        
        
        # small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25,interpolation = cv2.INTER_CUBIC)
        rgb_small_frame = frame[:, :, ::-1]
        toa_y = 100
        stt = 0
        for i in hiendien_ten:
            stt+=1
            toa_y += 40
        
        if int(str(len(rutgon_)-int(len(hiendien_ten)))) == 0 :
            pass
        name_linkfile = str(time.localtime(time.time()).tm_mon)+'_'+str(time.localtime(time.time()).tm_mday) +'_'+str(time.localtime(time.time()).tm_year)
        vang_os = open('regular_review/hour' + str(time.localtime(time.time()).tm_hour)+'.txt/'+name_linkfile+'_vang'+'.txt','a+') #write ck
        vang_os.close()
        vang_os = open('regular_review/hour' + str(time.localtime(time.time()).tm_hour)+'.txt/'+name_linkfile+'_vang'+'.txt','r+') #write ck
        for i in rutgon_:
            if i in hiendien_ten:
                continue
            else:
                if i not in list(vang_os.read().split()):
                    file_vang.append(i)
        vang_os.close()
        
        vang_os = open('regular_review/hour' + str(time.localtime(time.time()).tm_hour)+'.txt/'+name_linkfile+'_vang'+'.txt','w+')
        vang_os.close()
        #write ck

        for i in file_vang:
            try:
                vang_os = open('regular_review/hour' + str(time.localtime(time.time()).tm_hour)+'.txt/'+name_linkfile+'_vang'+'.txt','r+')
                zzz = list(vang_os.read().split())
                vang_os.close()
            except:
                traceback.print_exc()
                continue
            if i not in zzz and i not in hiendien_ten:
                vang_os = open('regular_review/hour' + str(time.localtime(time.time()).tm_hour)+'.txt/'+name_linkfile+'_vang'+'.txt','a+') #write ck
                #if time.localtime(time.time()).tm_min==40:
                    #if time.localtime(time.time()).tm_sec==50:
                vang_os.write("\n"+i)
                vang_os.close()
            else:
                vang_os.close()

        letiennhat = open('hide_on_push/id.txt','w+')
        letiennhat.write('WELCOME TO')
        letiennhat.close()
        letiennhat = open('hide_on_push/name_id.txt','w+')
        letiennhat.write('WELCOME TO')
        letiennhat.close()
        letiennhat = open('hide_on_push/donvi.txt','w+')
        letiennhat.write('DUT')
        letiennhat.close()
        if os.path.exists('best/input_1.txt'):
            button = 1
        if button == 1:
            music("input.mp3")
            try:
                if not os.path.exists('best/input_1.txt'):
                    time.sleep(2)
                else:
                    open_fff = open('best/input_1.txt', 'r+')
                    val_input_2 = open_fff.read()
                    open_fff.close()
                    os.remove('best/input_1.txt')
                    text = val_input_2
            except:
                traceback.print_exc()
                passgit 
            music("bandasansang.mp3")
        
            dem_anh = 0
            while True:
                while True:
                    if dem_anh > 50:
                        dem_nhac += 1

                        music("success.mp3")
                        cv2.destroyAllWindows()
                        return main(argv, dem_nhac)  # return
                        break

                    image_ = image_queue.get()
                    cv2.imshow('add_person', image_)

                    if cv2.waitKey(1) & 0xff == ord('a'):
                        try:
                            if not os.path.exists('image_css/' + text):
                                os.mkdir('image_css/' + text)

                        except:
                            traceback.print_exc()
                            pass

                        if dem_anh == 0:
                            music("dunglabanroi.mp3")
                        cv2.imwrite('image_css/' + str(text) + '/' + 'classe' + str(dem_anh) + '.jpg', image_)
                        list_image_css.append('image_css/' + str(text) + '/classe' + str(dem_anh) + '.jpg')
                        dem_anh = dem_anh + 1

            del (video)
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            # face_locations = face_recognition.face_locations(rgb_small_frame, 2)
            # print(face_locations)
            # print('Processing frame')
        
            faces = detector.detect_faces(rgb_small_frame)
            boxes = [face['box'] for face in faces]
            face_locations = [(y,x+w,y+h,x) for (x,y,w,h) in boxes]
            # print('face_location: ', face_locations)
            # print(face_locations)
            
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            #start = time.time()
            name_linkfile = str(time.localtime(time.time()).tm_mon) +'_'+str(time.localtime(time.time()).tm_mday) +'_'+str(time.localtime(time.time()).tm_year)
            vang_os = open('regular_review/hour' + str(time.localtime(time.time()).tm_hour)+
                           '.txt/'+name_linkfile+'_vang'+'.txt','a+') #write ck
            vang_os.close()
            vang_os = open('regular_review/hour' + str(time.localtime(time.time()).tm_hour)+
                               '.txt/'+name_linkfile+'_vang'+'.txt','r+') #write ck

                   
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.3019542298956785)
                if bien_==0:
                    open_f = open('best/input.txt','w+')
                    val_input = open_f.write("HELLO MY FRIEND ")
                    open_f.close()
                if True not in matches:
                    bien_+=1
                    if bien_>=2:
                        open_f = open('best/input.txt','w+')
                        val_input = open_f.write("Do you want Input new friend ")
                        open_f.close()
                        if os.path.exists('best/input_1.txt'):
                            button=1
                        if button == 1:
                            music("input.mp3")
                            try:
                                if not os.path.exists('best/input_1.txt'):
                                    time.sleep(2)
                                else:
                                    open_fff = open('best/input_1.txt','r+')
                                    val_input_2 = open_fff.read()
                                    open_fff.close()
                                    os.remove('best/input_1.txt')
                                    text = val_input_2
                            except:
                                traceback.print_exc()
                                pass
                            music("bandasansang.mp3")
                            dem_anh = 0
                            while True:
                                while True:
                                    if dem_anh > 50:
                                        dem_nhac += 1
                                        
                                        music("success.mp3")
                                        cv2.destroyAllWindows()
                                        return main(argv,dem_nhac) # return
                                        break
            
                                    image_ = image_queue.get()
                                    cv2.imshow('add_person', image_)

                                    if cv2.waitKey(1) & 0xff == ord('a'):
                                        try:
                                            if not os.path.exists('image_css/'+text):
                                                os.mkdir('image_css/'+text)
                                        
                                        except:
                                            traceback.print_exc()
                                            pass
                                    
                                        if dem_anh==0:
                                            music("dunglabanroi.mp3")
                                        cv2.imwrite('image_css/'+str(text)+'/'+'classe'+str(dem_anh)+'.jpg',image_)
                                        list_image_css.append('image_css/'+str(text)+'/classe'+str(dem_anh)+'.jpg')
                                        dem_anh = dem_anh +1
                            del(cap)
                name = "Unknow"
                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)
                #problem realtime : regular review
                name_linkfile = str(time.localtime(time.time()).tm_mon)+'_'+str(time.localtime(time.time()).tm_mday)+'_'+str(time.localtime(time.time()).tm_year)
                
                # letiennhat = open('best/ten.txt','w+')
                try:
                    if int(name).isdigit():
                        letiennhat = open('hide_on_push/id.txt','w+')
                        letiennhat.write(name)
                        letiennhat.close()
                    else:
                        letiennhat = open('hide_on_push/name_id.txt','w+')
                        letiennhat.write(splitname(name))
                        letiennhat.close()
                        letiennhat = open('hide_on_push/id.txt','w+')
                        letiennhat.write(name)
                        letiennhat.close()
                except:
                    traceback.print_exc()
                    if name != "Unknow":
                        letiennhat = open('hide_on_push/name_id.txt','w+')
                        letiennhat.write(splitname(name))
                        letiennhat.close()
                        letiennhat = open('hide_on_push/id.txt','w+')
                        letiennhat.write(name)
                        letiennhat.close()
                try:
                    LeTienNhat= open('hide_on_push/html3.html','w+')
                    LeTienNhat.write('<meta http-equiv="refresh" content="2"><img src="scv/'+str(name)+'.jpg" alt="Anh dai dien" width="100%" height="100%">')
                    LeTienNhat.close()
                except:
                    pass
                if name != "Unknow":
                    time.sleep(2)
                
                face_locations_count = face_recognition.face_locations(frame)
                
                if len(face_locations_count)>=2:
                    letiennhat = open('best/ten.txt','w+')
                    letiennhat.write("EVERYONE")
                    letiennhat.close()
                if name != "Unknow":
                    unknow_count=0
                    name_file = 'regular_review/hour'+str(time.localtime(time.time()).tm_hour)+'.txt'
                    open_file = open(name_file+'/'+name_linkfile+'.txt','a+')#create if not have
                    open_file.close()
                    
                    open_file = open(name_file+'/'+name_linkfile+'.txt','r+')#thang_ngay_nam.txt
                    #open_file1 = open(name_file+'/1.txt','a+') #check tinh thuong xuyen
                    time_os = open('regular_review/hour'+str(time.localtime(time.time()).tm_hour)+'.txt/'+name_linkfile+'.txt','r+')
                    hiendien_ten = list(time_os.read().split())
                    if name in rutgon_:
                        if name not in hiendien_ten:
                            hiendien += 1
                    
                    if name not in list(open_file.read().split()):
                        database = pymysql.connect(host='localhost',
                                                   user='be',
                                                   password='blueeyes',

                                                   )
                        cursor = database.cursor()
                        image_64 = name+str(time.localtime(time.time()).tm_mday)+str(time.localtime(time.time()).tm_mon)+str(time.localtime(time.time()).tm_year)+str(time.localtime(time.time()).tm_hour)+'readbase'
                        cv2.imwrite('status_sv/'+image_64+'.jpg',frame)
                        # time.sleep(0.5)
                        #time.sleep(2
                        image_base = open('status_sv/'+image_64+'.jpg', 'rb')
                        
                        imread = image_base.read()
                        image_64 = base64.encodebytes(imread)

                        # print(image_64)
                        # print(type(image_64))
                        image_base.close()
                        try:
                            os.remove('status_sv/'+image_64+'.jpg')
                        except:
                            pass
                        # test_base = open('test64.txt','w+')
                        # test_base.write(image_64.decode('utf-8'))
                        # test_base.close()

                        # image_64=image_64.decode("utf-8")
                        # print(type(image_64))

                        cursor.execute("use mysqldb1")
                        # cursor.execute("insert into b values({})".format(image_64))
                        query = """insert into realtime_0 values(%s,%s,%s,%s,%s,%s)"""
                        query_1 = """insert into realtime_0 select %s,%s,%s,%s,%s,%s,hovaten,donvi,ngaysinh,gioitinh,\
                        chucvu,trinhdo,hocham from manager where maso = %s"""
                        value_query = name,str(time.localtime(time.time()).tm_mday),str(time.localtime(time.time()).tm_mon),str(time.localtime(time.time()).tm_year),str(time.localtime(time.time()).tm_hour),image_64,name
                        cursor.execute(query_1,value_query)
                        database.commit()
                        cursor.close()
                        database.close()
                        bien_vang=1
                        open_file.close()
                        open_file = open(name_file+'/'+name_linkfile+'.txt','a+')#thang_ngay_nam.txt
                        open_file.write('\n'+str(name))
                        open_file.close()
                elif name =="Unknow":
                    unknow_count +=1
                    if unknow_count >=10:
                        unknow_count =0
                        print("warning")
                        image_64 = name+str(time.localtime(time.time()).tm_mday)+str(time.localtime(time.time()).tm_mon)+str(time.localtime(time.time()).tm_year)+str(time.localtime(time.time()).tm_hour)+'readbase'
                        cv2.imwrite('status_sv/'+image_64+'.jpg',frame)
                        # time.sleep(0.5)
                        #time.sleep(2
                        image_base = open('status_sv/'+image_64+'.jpg', 'rb')
                        
                        imread = image_base.read()
                        image_64 = base64.encodebytes(imread)
                        

                        # print(image_64)
                        # print(type(image_64))
                        image_base.close()
                        try:
                            os.remove('status_sv/'+image_64+'.jpg')
                        except:
                            pass
                        try:
                            print(111)
                            database1 = pymysql.connect(host='localhost',
                                                    user='be',
                                                    password='blueeyes',

                                                    )
                            print(222)
                            cursor1 = database1.cursor()
                            print(333)
                            cursor1.execute("use mysqldb1")
                            query_3="""insert into realtime_0(id,ngay,thang,nam,gio,url,hovaten,donvi,ngaysinh,gioitinh,chucvu,trinhdo,hocham) values(0,%s,%s,%s,%s,%s,'unknown','unknown',0,0,0,0,0)"""
                            values_2 = str(time.localtime(time.time()).tm_mday),str(time.localtime(time.time()).tm_mon),str(time.localtime(time.time()).tm_year),str(time.localtime(time.time()).tm_hour),image_64
                            cursor1.execute(query_3,values_2)
                            database1.commit()
                            cursor1.close()
                            database1.close()
                        except:
                            pass
                time_os = open('regular_review/hour'+str(time.localtime(time.time()).tm_hour)+'.txt/'+name_linkfile+'.txt','a+')
                time_os.close()
                time_os = open('regular_review/hour'+str(time.localtime(time.time()).tm_hour)+'.txt/'+name_linkfile+'.txt','r+')
                try:
                    if name in list(time_os.read().split()):
                        thuongxuyen_ = 1
                    
                    else :
                        thuongxuyen_ = 0
                    
                except:
                    traceback.print_exc()
                    pass
                time_os.close()
                    
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5*2
        thick = 1
        glow = 2 * thick
        
        # Color settings
        black = (0, 0, 0)
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        red = (0, 255, 255)
        #cv2.putText(frame, localtime, (800, 50), font, scale, red, glow)
        if name != "Unknow":
            if name in k:
                bien=1
            else:
                bien=0
        else:
            if bien_==10 :
                bien=0
                #bien_=0
            else:
                bien=1

    
        for i in k:
            if i == "Unknow":
                k.remove("Unknow")
        ls=[]
        for i in k:
            if i not in ls:
                n.append(i)

        k=ls
        process_this_frame = not process_this_frame

        time_local = time.localtime(time.time())   #truy xuat thong tin time

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            #print(distance_cal(4.0,870,int(right-left))*2.54)
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            # problem : nhieu nguoi va 1 nguoi .
            #k=[name]
            k.append(name)
            n=[]
            for i in k:
                if i not in n:
                    n.append(i)
            #print(n)
            if len(n)>=2 :
                    text_signal = open("best/signal.txt",'w+')
                    text_signal.write("1")
                    text_signal.close()
                    # text_signal = open("best/ten.txt",'w+')
                    # text_signal.write("EVERYONE")
                    # text_signal.close()
                    #time.sleep(1)
            elif(len(n)<2):
                text_signal = open("best/signal.txt","w+")
                text_signal.write("0")
                text_signal.close()
                #time.sleep(1)
            if name!="Unknow" :
                if len(n)>=2:
                    #print(len(n))
                    if bien==0:
                        print("chao cac ban")
                    for i in n:
                        if i not in hiendien_ten and (i != "Unknow"):
                            pygame.mixer.init()
                            if time_local.tm_hour in range(12):
                                pygame.mixer.music.load("goodmorningeveryone!.mp3")
                            elif time_local.tm_hour in range(12,17):
                                pygame.mixer.music.load("goodafternooneveryone.mp3")
                            else:
                                pygame.mixer.music.load("goodeveningeveryone.mp3")
                            pygame.mixer.music.play()
                            break
                else:
                    if name not in hiendien_ten:
                        if bien==0:
                            print("Chao ban "+name)
                        pygame.mixer.init()
                        if time_local.tm_hour in range(12):
                            pygame.mixer.music.load("goodmorning_.mp3")
                        elif time_local.tm_hour in range(12,17):
                            pygame.mixer.music.load("goodafternoon.mp3")
                        else:
                            pygame.mixer.music.load("goodevening.mp3")
                            #if thuongxuyen_ == 0 :
                        pygame.mixer.music.play()
            else :
                if bien_ >5:
                    cv2.putText(frame, "WARNING UNKNOW", (400,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,179,120),4,cv2.LINE_AA)
                if bien==0:
                    music("nicetomeetyou.mp3")
                    print("Canh bao nguoi la")
            cv2.putText(frame,name, (left + 6, bottom - 6), font, 1.0, (000, 255, 255), 1)
            if name not in ["TranQuangHuy","NguyenLuongQuang","LeTienNhat","LaTanDat"] and name!="Unknow":
            	cv2.putText(frame,"Kinh Chao Thay/Co : ",(550,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,70,100),4);cv2.putText(frame,str(name),(550,175),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,100),4)
            	if a =="happiness":
            		cv2.putText(frame, "He Thong Phat hien",(50,600),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,100),4)
            		cv2.putText(frame, "Thay/Co dang vui",(90,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,100),4)
            	#elif a=="neutral":
            		#cv2.putText(frame, "He Thong Phat hien",(50,600),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,110),4)
            		#cv2.putText(frame, "Thay/Co dang bin",(90,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,110),4)
            elif name in ["TranQuangHuy","NguyenLuongQuang","LeTienNhat","LaTanDat"]:
            	if a =="happiness":
            		cv2.putText(frame, "He Thong Phat hien",(50,600),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,110),4)
            		cv2.putText(frame, "Ban dang vui",(90,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,110),4)
            	#elif a=="neutral":
            		#cv2.putText(frame, "He Thong Phat hien",(50,600),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,110),4)
            		#cv2.putText(frame, "Ban dang buon",(90,650),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,110),4)
            else:
            	pass
        cv2.imshow(sourceName, frame)
        cap.write(frame)
        if paused:
            key = cv2.waitKey(0)
        else:
            end = datetime.now()
            delta = (end - start)
            if fps != 0:
                delay = 0.001#int(max(1, ((1 / fps) - delta.total_seconds()) * 100))
            else:
                delay = 1
            key = cv2.waitKey(delay)
        if key == ord('q') or key == ord('Q') or key == 27:
        	cap.release()
        	break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
        elif args.source == 'video' and (key == ord('r') or key == ord('R')):
            frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif args.source == 'video' and paused and key == 2424832: # Left key
            frameNum -= 1
            if frameNum < 0:
                frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif args.source == 'video' and paused and key == 2555904: # Right key
            frameNum += 1
            if frameNum >= frameCount:
                frameNum = frameCount - 1
        elif args.source == 'video' and key == 2162688: # Pageup key
            frameNum -= (fps * 10)
            if frameNum < 0:
                frameNum = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif args.source == 'video' and key == 2228224: # Pagedown key
            frameNum += (fps * 10)
            if frameNum >= frameCount:
                frameNum = frameCount - 1
            video.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        elif key == 7340032: # F1
            showHelp(sourceName, frame.shape)

        if not paused:
            frameNum += 1
    video.release()
    cap.release()
    cv2.destroyAllWindows()

#---------------------------------------------
def drawInfo(frame, frameNum, frameCount, paused, fps, source):
    """
    Draws text info related to the given frame number into the frame image.

    Parameters
    ----------
    image: numpy.ndarray
        Image data where to draw the text info.
    frameNum: int
        Number of the frame of which to drawn the text info.
    frameCount: int
        Number total of frames in the video.
    paused: bool
        Indication if the video is paused or not.
    fps: int
        Frame rate (in frames per second) of the video for time calculation.
    source: str
        Source of the input images (either "video" or "cam").
    """

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    glow = 3 * thick

    # Color settings
    black = (0, 0, 0)
    yellow = (0, 255, 255)

    # Print the current frame number and timestamp
    if source == 'video':
        text = 'Frame: {:d}/{:d} {}'.format(frameNum, frameCount - 1,
                                            '(paused)' if paused else '')
    else:
        text = 'Frame: {:d} {}'.format(frameNum, '(paused)' if paused else '')
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = 5
    y = frame.shape[0] - 2 * size[1]
    cv2.putText(frame, text, (x, y), font, scale, black, glow)
    cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

    if source == 'video':
        timestamp = datetime.min + timedelta(seconds=(frameNum / fps))
        elapsedTime = datetime.strftime(timestamp, '%H:%M:%S')
        timestamp = datetime.min + timedelta(seconds=(frameCount / fps))
        totalTime = datetime.strftime(timestamp, '%H:%M:%S')

        text = 'Time: {}/{}'.format(elapsedTime, totalTime)
        size, _ = cv2.getTextSize(text, font, scale, thick)
        y = frame.shape[0] - 5
        cv2.putText(frame, text, (x, y), font, scale, black, glow)
        cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

    # Print the help message
    text = 'Press F1 for help'
    size, _ = cv2.getTextSize(text, font, scale, thick)
    x = frame.shape[1] - size[0] - 5
    y = frame.shape[0] - size[1] + 5
    cv2.putText(frame, text, (x, y), font, scale, black, glow)
    cv2.putText(frame, text, (x, y), font, scale, yellow, thick)

#---------------------------------------------
def showHelp(windowTitle, shape):
    """
    Displays an image with helping text.

    Parameters
    ----------
    windowTitle: str
        Title of the window where to display the help
    shape: tuple
        Height and width of the window to create the help image.
    """

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 1

    # Color settings
    black = (0, 0, 0)
    red = (0, 0, 255)

    # Create the background image
    image = np.ones((shape[0], shape[1], 3)) * 255

    # The help text is printed in one line per item in this list
    helpText = [
    'Controls:',
    '-----------------------------------------------',
    '[q] or [ESC]: quits from the application.',
    '[p]: toggles paused/playing the video/webcam input.',
    '[r]: restarts the video playback (video input only).',
    '[left/right arrow]: displays the previous/next frame (video input only).',
    '[page-up/down]: rewinds/fast forwards by 10 seconds (video input only).',
    ' ',
    ' ',
    'Press any key to close this window...'
    ]

    # Print the controls help text
    xCenter = image.shape[1] // 2
    yCenter = image.shape[0] // 2

    margin = 20 # between-lines margin in pixels
    textWidth = 0
    textHeight = margin * (len(helpText) - 1)
    lineHeight = 0
    for line in helpText:
        size, _ = cv2.getTextSize(line, font, scale, thick)
        textHeight += size[1]
        textWidth = size[0] if size[0] > textWidth else textWidth
        lineHeight = size[1] if size[1] > lineHeight else lineHeight

    x = xCenter - textWidth // 2
    y = yCenter - textHeight // 2

    for line in helpText:
        cv2.putText(image, line, (x, y), font, scale, black, thick * 3)
        cv2.putText(image, line, (x, y), font, scale, red, thick)
        y += margin + lineHeight

    # Show the image and wait for a key press
    cv2.imshow(windowTitle, image)
    cv2.waitKey(0)

if __name__ == '__main__':
    while True:
       main(sys.argv[1:])

