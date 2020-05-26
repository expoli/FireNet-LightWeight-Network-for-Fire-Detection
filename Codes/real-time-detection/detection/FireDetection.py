import time

import cv2
import numpy as np
import tensorflow as tf

from Preprocessor import CreateFilesPath
from Preprocessor import ModelLoader
from utils.alarm import mail as alarm_mail


class FireDetectioner:
    def __init__(self, IMG_SIZE=64, sleep_time=1, modelPath='', video_path='', gui_flag='',
                 window_name='Result'):
        self.IMG_SIZE = IMG_SIZE
        self.sleep_time = sleep_time
        self.video_path = CreateFilesPath.CreateFilesPath(video_path).create_path_list()
        self.gui_flag = gui_flag
        self.window_name = window_name
        self.model = ModelLoader.LoadModel(modelPath).load_saved_model()
        self.fire_list = []
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
        alarm_mail.mail(mail_text, mail_to)
        print("报警成功！")
        return 0

    def detection(self):
        model = self.model
        window_name = self.window_name
        for path in self.video_path:
            cap = cv2.VideoCapture(path)
            time.sleep(self.sleep_time)
            if cap.isOpened():
                while (1):
                    # try to get the first frame
                    rval, image = cap.read()
                    if (rval):
                        orig = image.copy()

                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
                        image = image.astype("float") / 255.0
                        image = tf.keras.preprocessing.image.img_to_array(image)
                        image = np.expand_dims(image, axis=0)

                        tic = time.time()
                        predictions = model.predict(image)
                        fire_prob = predictions[0][0] * 100
                        toc = time.time()

                        if self.epoch <= 100:
                            self.fire_list.append(fire_prob)
                            self.epoch += 1
                        elif np.mean(self.fire_list) >= 50:
                            print(np.mean(self.fire_list))
                            print("Fire! Alarm!!!")
                            self.epoch = 0

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
