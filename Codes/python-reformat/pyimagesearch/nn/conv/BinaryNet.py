import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten


class BinaryNet:
    @staticmethod
    def build(width, height, depth):
        model = tf.keras.models.Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=inputShape))
        model.add(AveragePooling2D())
        # model.add(Dropout(0.5))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(AveragePooling2D())
        # model.add(Dropout(0.5))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(AveragePooling2D())
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(units=128, activation='relu'))

        model.add(Dense(units=2, activation='softmax'))

        return model
