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


init_gpu()

# loading the stored model from file
model = load_model('my_model.h5')

cap = cv2.VideoCapture('Datasets/Our_Complete_Dataset_Video_and_extra_NoFire_Frames/fire/FireVid20.mp4')
time.sleep(2)

if cap.isOpened():  # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False


IMG_SIZE = 64
# IMG_SIZE = 224

#for i in range(2500):
#    cap.read()

while(1):

    rval, image = cap.read()
    if rval==True:
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
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.namedWindow("Output")
        cv2.imshow("Output", orig)

        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            break
    elif rval == False:
        break
end = time.time()

cap.release()
cv2.destroyAllWindows()
