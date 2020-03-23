import itertools
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


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


def create_test_image_data(DATADIR, IMG_SIZE):
    test_image_data = []
    CATEGORIES = os.listdir(DATADIR)

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        if 'NoFire' in category:
            class_num = 1
        elif 'Fire' in category:
            class_num = 0

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                test_image_data.append([new_array, class_num])  # add this to our test_image_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    return test_image_data


def random_shuffle_data(test_image_data):
    print(len(test_image_data))
    random.shuffle(test_image_data)
    return test_image_data


def create_test_labels(test_image_data, test_image_num=871):
    c = 0
    test_labels = np.zeros((test_image_num, 1))
    for sample in test_image_data:
        test_labels[c] = (sample[1])
        c += 1
    print(c)
    actual_labels = (test_labels.reshape(test_image_num, ))
    print(actual_labels.shape)
    actual_labels.astype(int)
    return actual_labels


def create_dataset(test_image_data, IMG_SIZE=64):
    X = []
    Y = []

    for features, label in test_image_data:
        X.append(features)
        Y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X = X / 255.0
    # X.shape[1:]

    Y = np.array(Y)
    return X, Y


def load_tf_h5_model(model_path='my_model.h5'):
    model = tf.keras.models.load_model('my_model.h5')
    return model


def predicte_labels(X, model, test_image_num):
    predicted_labels = model.predict_classes(X)
    predicted_labels = (predicted_labels.reshape(test_image_num, ))
    predicted_labels.astype(int)

    return predicted_labels


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return 0


def begain_compose(actual_labels, predicted_labels):
    cm = confusion_matrix(actual_labels, predicted_labels)
    # test_batches.class_indices
    cm_plot_labels = ['Fire', 'No Fire']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    print("tp" + ' ' + str(tp))
    print("fn" + ' ' + str(fn))
    print("fp" + ' ' + str(fp))
    print("tn" + ' ' + str(tn))

    Recall = tp / (tp + fn)
    Precision = tp / (tp + fp)
    f_measure = 2 * ((Precision * Recall) / (Precision + Recall))

    print('Precision=', Precision, 'Recall=', Recall, 'f_measure=', f_measure)

    return 0

def dispaly_model_summary(model):
    # 显示模型的结构
    model.summary()
    return 0

def evaluate_model(X, Y, model):
    result = model.evaluate(X, Y)

    return result


if __name__ == '__main__':
    init_gpu()
    test_image_data = create_test_image_data(DATADIR='Datasets/Test_Dataset1__Our_Own_Dataset', IMG_SIZE=64)
    shuffled_test_image_data = random_shuffle_data(test_image_data)
    test_image_num = len(shuffled_test_image_data)
    actual_labels = create_test_labels(shuffled_test_image_data, test_image_num=test_image_num)
    X, Y = create_dataset(test_image_data=shuffled_test_image_data, IMG_SIZE=64)
    model = load_tf_h5_model(model_path='my_model.h5')
    dispaly_model_summary(model)
    predicted_labels = predicte_labels(X=X, model=model, test_image_num=test_image_num)
    begain_compose(actual_labels=actual_labels, predicted_labels=predicted_labels)
    print(evaluate_model(X, Y, model))
