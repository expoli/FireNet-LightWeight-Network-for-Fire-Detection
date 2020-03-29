import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return 0

def load_saved_model(model_path='my_model.h5'):
    # loading the stored model from file
    model = load_model(model_path)
    return model


def capture_video(video_path='Datasets/Our_Complete_Dataset_Video_and_extra_NoFire_Frames/fire/FireVid20.mp4'):
    cap = cv2.VideoCapture(video_path)
    time.sleep(2)

    if cap.isOpened():  # try to get the first frame
        rval, frame = cap.read()
    else:
        rval = False

    return cap, rval


def file_detection(cap, IMG_SIZE, model, window_name="Output"):
    while (1):
        rval, image = cap.read()
        if rval == True:
            orig = image.copy()

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            tic = time.time()
            fire_prob = model.predict(image)[0][0] * 100
            toc = time.time()
            print("Time taken = ", toc - tic)
            print("FPS: ", 1 / np.float64(toc - tic))
            print("Fire Probability: ", fire_prob)
            print("Predictions: ", model.predict(image))
            print(image.shape)

            label = "Fire Probability: " + str(fire_prob)
            fps_label = "FPS: " + str(1 / np.float64(toc - tic))
            cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(orig, fps_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.namedWindow(window_name)
            cv2.imshow(window_name, orig)

            key = cv2.waitKey(10)
            if key == 27:  # exit on ESC
                cap.release()
                cv2.destroyAllWindows()
                break
        elif rval == False:
            break


if __name__ == '__main__':
    init_gpu()
    model = load_saved_model()
    cap, rval = capture_video()
    file_detection(cap=cap, IMG_SIZE=64, model=model, window_name="Result")
