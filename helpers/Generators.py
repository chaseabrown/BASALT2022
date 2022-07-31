import numpy as np
import PIL
from random import shuffle
import random
import keras
import cv2
import math
from keras.preprocessing.image import load_img

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


class DepthEstGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(768, 1024), n_channels=3, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, image_path, depth_map, mask):
        """Load input and target image."""

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0

        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def data_generation(self, batch):

        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["depth"][batch_id],
                self.data["mask"][batch_id],
            )

        return x, y

class SegmentationGenerator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="RGB")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y

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
