import time
from multiprocessing import Process

import cv2
import numpy as np
import tensorflow as tf

from Preprocessor import CreateFilesPath
from Preprocessor import ModelLoader
from utils.alarm.mail import mail as alarm_mail
from utils.alarm.private_info import *


class FireDetectioner:
    def __init__(self, IMG_SIZE=64, modelPath='', video_path='', gui_flag='', window_name='Result', DEVICE_ID="0000001",
                 max_list_num=200, rtsp_url=''):
        self.DEVICE_ID = DEVICE_ID
        self.IMG_SIZE = IMG_SIZE
        self.video_path = video_path
        self.gui_flag = gui_flag
        self.window_name = window_name
        self.model = ModelLoader.LoadModel(modelPath).load_saved_model()
        self.max_list_num = max_list_num
        self.fire_list = []
        self.rtsp_url = rtsp_url
        self.epoch = 0

    def textOuter(self, tic, toc, fire_prob, predictions):
        print("Time taken = ", toc - tic)
        print("FPS: ", 1 / np.float64(toc - tic))
        print("Fire Probability: ", fire_prob)
        print("Predictions: ", predictions)
        return 0

    def guiOutputer(self, orig, path, tic, toc, fire_prob, window_name):
        label = "Fire Probability: " + str(fire_prob)
        fps_label = "FPS: " + str(1 / np.float64(toc - tic))
        cv2.putText(orig, path, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(orig, fps_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(winname=window_name, width=1080, height=720)
        cv2.imshow(winname=window_name, mat=orig)
        return 0

    def mailAlarm(self, mail_text, mail_to):
        alarm_mail(mail_text, mail_to)
        print("报警成功！")
        return 0

    def detection(self):
        if self.rtsp_url != '':
            self.normalDetection()
        else:
            self.videoDetection()
        
    def normalDetection(self):
        model = self.model
        cap = cv2.VideoCapture(self.rtsp_url)
        if cap.isOpened():
            while (1):
                # try to get the first frame
                rval, image = cap.read()
                if (rval):
                    orig = image.copy()

                    # 数据预处理
                    image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
                    image = image.astype("float") / 255.0
                    image = tf.keras.preprocessing.image.img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    # 帧率计算
                    tic = time.time()
                    predictions = model.predict(image)
                    fire_prob = predictions[0][0] * 100
                    toc = time.time()
                    # 火情判断
                    if self.epoch <= self.max_list_num:
                        self.fire_list.append(fire_prob)
                        self.epoch += 1
                    elif np.mean(self.fire_list) >= 50:
                        print(np.mean(self.fire_list))
                        print("Fire! Alarm!!!")
                        try:
                            mail_process = Process(target=self.mailAlarm,args=("Fire! Alarm!!!\n" +
                                        "Device ID:" + self.DEVICE_ID +
                                        "\nTime:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), MAIL_TO))
                            mail_process.start()
                        except:
                            print("Error: 无法启动邮件线程")
                        self.epoch = 0
                    # gui 窗口，选择
                    if self.gui_flag == '1':
                        self.guiOutputer(orig, '', tic, toc, fire_prob, self.window_name)
                    else:
                        self.textOuter(tic, toc, fire_prob, predictions)

                    key = cv2.waitKey(10)
                    if key == 27:  # exit on ESC
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                else:
                    rval = False
                    break
        else:
            print("URL Error! break!")
            return -1
        return 0


    def videoDetection(self):
        model = self.model
        video_path = CreateFilesPath.CreateFilesPath(self.video_path).create_path_list()
        for path in video_path:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                while (1):
                    # try to get the first frame
                    rval, image = cap.read()
                    if (rval):
                        orig = image.copy()

                        # 数据预处理
                        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
                        image = image.astype("float") / 255.0
                        image = tf.keras.preprocessing.image.img_to_array(image)
                        image = np.expand_dims(image, axis=0)
                        # 帧率计算
                        tic = time.time()
                        predictions = model.predict(image)
                        fire_prob = predictions[0][0] * 100
                        toc = time.time()
                        # 火情判断
                        if self.epoch <= self.max_list_num:
                            self.fire_list.append(fire_prob)
                            self.epoch += 1
                        elif np.mean(self.fire_list) >= 50:
                            print(np.mean(self.fire_list))
                            print("Fire! Alarm!!!")
                            try:
                                mail_process = Process(target=self.mailAlarm,args=("Fire! Alarm!!!\n" +
                                           "Device ID:" + self.DEVICE_ID +
                                           "\nTime:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), MAIL_TO))
                                mail_process.start()
                            except:
                                print("Error: 无法启动邮件线程")
                            self.epoch = 0
                        # gui 窗口，选择
                        if self.gui_flag == '1':
                            self.guiOutputer(orig, path, tic, toc, fire_prob, self.window_name)
                        else:
                            self.textOuter(tic, toc, fire_prob, predictions)

                        key = cv2.waitKey(10)
                        if key == 27:  # exit on ESC
                            cap.release()
                            cv2.destroyAllWindows()
                            break
                    else:
                        rval = False
                        break
            else:
                print("Error! break!")
                break


        return 0
