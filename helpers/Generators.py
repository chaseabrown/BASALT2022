import numpy as np
import PIL
from random import shuffle
import random
import keras
import cv2
import math


def getFrames(videoPath, startFrame, numFrames, COLOR):
    cap = cv2.VideoCapture(videoPath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(1,startFrame)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = []

    fc = startFrame
    ret = True

    counter = numFrames-1
    while (fc < startFrame + numFrames):
        
        ret, frame = cap.read()
        if COLOR:
          buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
          buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fc += 1
        counter -= 1

    cap.release()
    return buf

class Generator2Images(keras.utils.Sequence):
    
    def __init__(self, images, labels,
                 batch_size,
                 inputShape=(640, 360, 3),
                 shuffle=True,
                 COLOR=True):

        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.shuffle = shuffle
        self.imageSize = inputShape[0], inputShape[1]
        self.COLOR = COLOR
        
        self.n = len(self.labels)

    def __get_input(self, image):
        
        image.thumbnail(self.imageSize, PIL.Image.ANTIALIAS)
        image = np.array(image)
        image = image.astype('float32')

        return image/255.
    
    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.images, self.labels))
            random.shuffle(temp)
            res1, res2 = zip(*temp)
            # res1 and res2 come out as tuples, and so must be converted to lists.
            self.images, self.labels = list(res1), list(res2)
    
    def __getitem__(self, index):

        imageBatch = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        labelBatch = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(imageBatch, labelBatch)        
        return X, y
    
    def test_getitem(self, index):
        return self.__getitem__(index)
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __get_output(self, startImages, endImages, labelBatch):
        
        X1 = np.array(startImages, np.float32)
        X2 = np.array(endImages, np.float32)

        Y = np.array(list(map(int,labelBatch)), np.int64)
        
        return X1, X2, Y

    def __get_data(self, imageBatch, labelBatch):
        # Generates data containing batch_size samples
        
        startImages = []
        endImages = []
        for path, startFrame in imageBatch:
            frames = getFrames(path, startFrame, 2, self.COLOR)
            startImages.append(self.__get_input(PIL.Image.fromarray(frames[0])))
            endImages.append(self.__get_input(PIL.Image.fromarray(frames[1])))
        
        X1, X2, Y = self.__get_output(startImages, endImages, labelBatch)

        return tuple([[X1, X2], Y])


class GeneratorStartImage(keras.utils.Sequence):
    
    def __init__(self, images, labels,
                 batch_size,
                 inputShape=(640, 360, 3),
                 shuffle=True,
                 COLOR=True):

        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.shuffle = shuffle
        self.imageSize = inputShape[0], inputShape[1]
        self.COLOR = COLOR
        
        self.n = len(self.labels)

        
    def __get_input(self, image):
        
        image.thumbnail(self.imageSize, PIL.Image.ANTIALIAS)
        image = np.array(image)
        image = image.astype('float32')

        return image/255.
    
    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.images, self.labels))
            random.shuffle(temp)
            res1, res2 = zip(*temp)
            # res1 and res2 come out as tuples, and so must be converted to lists.
            self.images, self.labels = list(res1), list(res2)
    
    def __getitem__(self, index):

        imageBatch = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        labelBatch = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(imageBatch, labelBatch)        
        return X, y
    
    def test_getitem(self, index):
        return self.__getitem__(index)
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __get_output(self, startImages, labelBatch):
        
        X1 = np.array(startImages, np.float32)

        Y = np.array(list(map(int,labelBatch)), np.int64)
        
        return X1, Y

    def __get_data(self, imageBatch, labelBatch):
        # Generates data containing batch_size samples
        
        startImages = []
        for path, startFrame in imageBatch:
            frames = getFrames(path, startFrame, 1, self.COLOR)
            startImages.append(self.__get_input(PIL.Image.fromarray(frames[0])))
        
        X1, Y = self.__get_output(startImages, labelBatch)

        return tuple([X1, Y])

class GeneratorEndImage(keras.utils.Sequence):
    
    def __init__(self, images, labels,
                 batch_size,
                 inputShape=(640, 360, 3),
                 shuffle=True,
                 COLOR=True):

        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.shuffle = shuffle
        self.imageSize = inputShape[0], inputShape[1]
        self.COLOR = COLOR

        self.n = len(self.labels)

        
    def __get_input(self, image):
        
        image.thumbnail(self.imageSize, PIL.Image.ANTIALIAS)
        image = np.array(image)
        image = image.astype('float32')

        return image/255.
    
    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.images, self.labels))
            random.shuffle(temp)
            res1, res2 = zip(*temp)
            # res1 and res2 come out as tuples, and so must be converted to lists.
            self.images, self.labels = list(res1), list(res2)
    
    def __getitem__(self, index):

        imageBatch = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        labelBatch = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(imageBatch, labelBatch)        
        return X, y
    
    def test_getitem(self, index):
        return self.__getitem__(index)
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __get_output(self, startImages, labelBatch):
        
        X1 = np.array(startImages, np.float32)

        Y = np.array(list(map(int,labelBatch)), np.int64)
        
        return X1, Y

    def __get_data(self, imageBatch, labelBatch):
        # Generates data containing batch_size samples
        
        startImages = []
        for path, startFrame in imageBatch:
            frames = getFrames(path, startFrame, 1, self.COLOR)
            startImages.append(self.__get_input(PIL.Image.fromarray(frames[0])))
        
        X1, Y = self.__get_output(startImages, labelBatch)

        return tuple([X1, Y])
