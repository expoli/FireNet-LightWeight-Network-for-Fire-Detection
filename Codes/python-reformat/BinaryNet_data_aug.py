import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from projectconfig import pathconfig
from pyimagesearch.datasets import SimpleDatasetLoader as SDL
from pyimagesearch.nn.conv import BinaryNet as BinNet
from pyimagesearch.preprocessing import AspectAwarePreprocessor as AAP
from pyimagesearch.preprocessing import ImageMove as IM
from pyimagesearch.preprocessing import ImageToArrayPreprocessor as IAP
from tfsettings.gpu import InitGpu
from tfsettings.traincallbacks import modelcheckpoint as TF_CB_chcekpoint
from tfsettings.traincallbacks import modelsaver as TF_CB_mdoel_saver
from tfsettings.traincallbacks import tensorboard as TF_CB_tensorboard

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
args = vars(ap.parse_args())

print('[INFO] initing gpu.....')
gpu = InitGpu.InitGpu()
gpu.init()

print('[INFO] moving image to label folder.....')
im = IM.MoveImageToLabel(dataPath=args['dataset'])
im.makeFolder()
im.move()

print("[INFO] loading images...")
imagePaths = [x for x in list(paths.list_images(args['dataset'])) if x.split(os.path.sep)[-2] != 'jpg']
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AAP.AspectAwarePreprocesser(64, 64)
iap = IAP.ImageToArrayPreprocess()

sdl = SDL.SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=43)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = BinNet.BinaryNet.build(width=64, height=64, depth=3)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network...")
# 构建回调函数
cb_tensorboard = TF_CB_tensorboard.TensorboardCallBack()
cb_mdoel_saver = TF_CB_mdoel_saver.modelsaver()
cb_chcekpoint = TF_CB_chcekpoint.ModelCheckpointCallBack()

epochs=600
H = model.fit(aug.flow(trainX, trainY, batch_size=32),
              validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
              epochs=epochs, verbose=1,
              callbacks=[cb_tensorboard.build_cb(), cb_chcekpoint.build_cb()])
model.save(filepath=pathconfig.get_model_save_path())
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=classNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
