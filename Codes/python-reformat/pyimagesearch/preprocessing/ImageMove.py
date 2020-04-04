# encoding:utf-8
import os
from glob import glob

from PIL import Image


class MoveImageToLabel:
    def __init__(self, dataPath):
        self.dataPath = dataPath

    def makeFolder(self):
        for i in range(2):
            foldername = self.dataPath + "/{0}".format(str(i))
            if not os.path.isdir(foldername):
                os.makedirs(foldername)

    def move(self):
        for imageName in glob(self.dataPath + "/jpg/*.jpg"):
            imageNum = imageName.split(".")[0][-4:]
            a = int(imageNum) // 80
            b = int(imageNum) % 80
            if b == 0:
                fl = self.dataPath + "/{0}/image_{1}.jpg".format(str(a - 1), imageNum)
                newimg = Image.open(imageName)
                newimg.save(fl)
            else:
                fl = self.dataPath + "/{0}/image_{1}.jpg".format(str(a), imageNum)
                newimg = Image.open(imageName)
                newimg.save(fl)
