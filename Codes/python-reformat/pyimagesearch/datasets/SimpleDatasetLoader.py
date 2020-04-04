import os

import cv2
import numpy as np


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for preprocessor in self.preprocessors:
                    try:
                        image.shape[:2]
                    except AttributeError as e:
                        print(imagePath)
                        continue
                    image = preprocessor.preprocess(image)

            data.append(image)
            labels.append(label)

            # print(imagePath)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {0}/{1}".format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))
