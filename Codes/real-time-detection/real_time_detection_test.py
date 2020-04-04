import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm


def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            return True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            return False


def load_saved_model(model_path='my_model.h5'):
    # loading the stored model from file
    model = load_model(model_path)
    return model


def create_video_files_path(CATEGORIES=['fire', 'NoFire'],
                            DATADIR='Datasets/Our_Complete_Dataset_Video_and_extra_NoFire_Frames'):
    video_files_path = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                video_files_path.append(os.path.join(path, img))  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
    return video_files_path


def file_detection(video_path, IMG_SIZE=64, saved_model_path='', window_name="Output"):
    model = load_saved_model(saved_model_path)
    for path in video_path:
        cap = cv2.VideoCapture(path)
        time.sleep(1)
        if cap.isOpened():
            while (1):
                # try to get the first frame
                rval, image = cap.read()
                if (rval):
                    orig = image.copy()

                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    image = image.astype("float") / 255.0
                    image = tf.keras.preprocessing.image.img_to_array(image)
                    image = np.expand_dims(image, axis=0)

                    tic = time.time()
                    fire_prob = model.predict(image)[0][0] * 100
                    toc = time.time()
                    # print("Time taken = ", toc - tic)
                    # print("FPS: ", 1 / np.float64(toc - tic))
                    # print("Fire Probability: ", fire_prob)
                    # print("Predictions: ", model.predict(image))
                    # print(image.shape)

                    label = "Fire Probability: " + str(fire_prob)
                    fps_label = "FPS: " + str(1 / np.float64(toc - tic))
                    cv2.putText(orig, path, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(orig, fps_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.namedWindow(window_name, 0)
                    cv2.resizeWindow(winname=window_name, width=1080, height=720)
                    cv2.imshow(winname=window_name, mat=orig)

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


if __name__ == '__main__':
    if init_gpu():
        video_files_path = create_video_files_path()
        # model = load_saved_model()
        file_detection(video_files_path, saved_model_path='result/train05/my_new_model_new_datasets.h5', IMG_SIZE=64,
                       window_name="Result")
    else:
        print('Error!')
