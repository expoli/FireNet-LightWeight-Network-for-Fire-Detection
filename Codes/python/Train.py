from __future__ import print_function

import os
import random

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tqdm import tqdm

# 设置 tnsorflow 的日志级别为 warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# 使用GPU进行计算的GPU初始化代码；如果不使用则会出现cudNN无法使用的报错

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


def create_training_data(CATEGORIES=['Fire', 'NoFire'], DATADIR='Datasets/Training Dataset', IMG_SIZE=64):
    training_data = []
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=C 1=O

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    return training_data


def shuffle_data(training_data):
    print(len(training_data))
    random.shuffle(training_data)
    for sample in training_data[:10]:
        print(sample[1])
    return training_data


def create_dataset(training_data, IMG_SIZE=64):
    X = []
    Y = []

    for features, label in training_data:
        X.append(features)
        Y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X = X / 255.0
    # X.shape[1:]

    Y = np.array(Y)
    return X, Y


# # set up image augmentation
# from keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
#     rotation_range=15,
#     horizontal_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1
#     #zoom_range=0.3
#     )
# datagen.fit(X)


def create_model(X, Y):
    model = tf.keras.models.Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
    model.add(AveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=128, activation='relu'))

    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def fit_and_save_model(X, Y, model, epochs=100,
                       checkpoint_path="./result/hdf5_files/weights.{epoch:02d}-{val_loss:.2f}.hdf5"):
    # 创建一个基本的模型实例

    # 显示模型的结构
    model.summary()

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 创建一个保存模型权重的回调
    # EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                  min_delta=0.2,
    #                                  patience=0,
    #                                  verbose=0,
    #                                  mode='auto',
    #                                  baseline=None,
    #                                  restore_best_weights=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./result/tensorboard_logs',
                                                          histogram_freq=0,
                                                          batch_size=32,
                                                          write_graph=True,
                                                          write_grads=False,
                                                          write_images=False,
                                                          embeddings_freq=0,
                                                          embeddings_layer_names=None,
                                                          embeddings_metadata=None,
                                                          embeddings_data=None,
                                                          update_freq='epoch')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     monitor='val_los',
                                                     verbose=0,
                                                     save_best_only=False,
                                                     save_weights_only=False,
                                                     mode='auto',
                                                     period=1)

    history = model.fit(X,
                        Y,
                        batch_size=32,
                        epochs=epochs,
                        validation_split=0.3,
                        callbacks=[cp_callback, tensorboard_callback],
                        )

    # 将整个模型保存为HDF5文件
    model.save('my_model.h5')

    return history


def load_model_file(h5_model_file_dir='my_model.h5'):
    model = tf.keras.models.load_model(h5_model_file_dir)
    if model == None:
        print("Load h5 model file fail!")
    else:
        print("Reload model file success!")

    return model


def plot_function(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # 'loss'
    # 'accuracy'
    return 0


if __name__ == '__main__':
    init_gpu()
    training_data = create_training_data()
    shuffled_data = shuffle_data(training_data)
    X, Y = create_dataset(shuffled_data)
    model = create_model(X, Y)
    # model = load_model_file('my_model.h5')
    history = fit_and_save_model(X, Y, model=model, epochs=1000)

    plot_function(history=history)
