#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:08:47 2022

@author: chasebrown
"""

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import cv2
import json
from tqdm import tqdm

def getFrames(videoPath):
    cap = cv2.VideoCapture(videoPath)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoFPS = int(cap.get(cv2.CAP_PROP_FPS))

    buf = np.empty((
        frameCount,
        frameHeight,
        frameWidth,
        3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    videoArray = buf
    return videoArray, frameWidth, frameHeight

def getMoves(movesPath):
    with open(movesPath, "r") as json_file:
        listOfMoves = []
        for line in json_file.readlines():
            try:
                moves = json.loads(line.replace(":", '":').replace("{", '{"').replace("], ", '], "'))
                for key in moves.keys():
                    if key == "camera":
                        clean = str(moves[key]).replace("[", "").replace("]", "").split(", ")
                        moves[key] = [float(clean[0]), float(clean[1])]
                    else:
                        moves[key] = int(str(moves[key]).replace("[", "").replace("]", ""))
                listOfMoves.append(moves)
            except Exception as e:
                print(e)
    return listOfMoves

def gatherData(dataPathList):
    samples = []
    for dataPath in dataPathList:
        for file in os.listdir(dataPath):
            if file.endswith(".txt"):
                try:
                    videoFile = os.listdir(dataPath + file.split('.')[0] + ".mp4/")[0]
                    sample = [dataPath + file.split('.')[0] + ".mp4/" + videoFile, dataPath + file]
                    samples.append(sample)
                except Exception as e:
                    print(file, e)

    return samples

def load_data(dim = (32,32,32), batch_size = 32, n_classes = 6, n_channels = 1, shuffle = True):
        dataPathList = ["../assets/datasets/Agent Moves/MineRLBasaltFindCave-v0/", 
                        "../assets/datasets/Agent Moves/MineRLBasaltBuildVillageHouse-v0/", 
                        "../assets/datasets/Agent Moves/MineRLBasaltCreateVillageAnimalPen-v0/", 
                        "../assets/datasets/Agent Moves/MineRLBasaltMakeWaterfall-v0/"]
        #path = "/content/drive/MyDrive/BASALT2022/basalt-2022-behavioural-cloning-baseline-main/video/"


        videoPaths, movePaths = gatherData(dataPathList) 
        
        params = {'dim': dim,
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': n_channels,
          'shuffle': shuffle}

        training_generator = DataGenerator(videoPaths['train'], movePaths, **params)
        validation_generator = DataGenerator(videoPaths['validation'], movePaths, **params)


        frames = []
        moves = {"index": [], "attack": [], "forward": [], "back": [], "left": [], "right": [], "jump": [], "sneak": [], "sprint": [], "use": [], "drop": [], "inventory": [], "hotbar": [], "camera1": [], "camera2": [], "ESC": []}
        inputShape = (64, 64, 3)
        for challenge in challenges[:2]:
            counter = 0
            for file in tqdm(os.listdir(path + challenge + "/")):
                if ".txt" in file:
                    if counter < 5:
                        try:
                            stamp = file.split(".")[0]
                            videoFile = os.listdir(path + challenge + "/" + stamp + ".mp4/")[0]
                            gameMoves = getMoves(path + challenge + "/" + file)
                            gameFrames, width, height = getFrames(path + challenge + "/" + stamp + ".mp4/" + videoFile)
                            inputShape = (width, height, 3)

                            for i in range(0, len(gameMoves)):

                                startImage = gameFrames[i].astype('float32')
                                startImage /= 255 
                                endImage = gameFrames[i+1].astype('float32')
                                endImage /= 255 
                                frames.append((startImage, endImage))
                                hotbar = 0
                                for key in gameMoves[i].keys():
                                    if key == "camera":
                                        moves["camera1"].append(gameMoves[i][key][0])
                                        moves["camera2"].append(gameMoves[i][key][1])
                                    elif "hotbar" in key:
                                        if gameMoves[i][key] == 1:
                                            hotbar = int(key.split(".")[1])
                                            
                                    else:
                                        moves[key].append(gameMoves[i][key])
                                moves["hotbar"].append(hotbar)
                                moves["index"].append(i)
                                counter += 1
                        except Exception as e:
                                print(e)
        
        print("Sampling...")

        xStartTrain = []
        xEndTrain = []
        yTrain = []
        xStartVal = []
        xEndVal = []
        yVal = []
        for i in range(0, len(trainImages)):
            xStartTrain.append(trainImages[i][0])
            xEndTrain.append(trainImages[i][1])
        yTrain = trainMoves

        for i in range(0, len(testImages)):
            xStartVal.append(testImages[i][0])
            xEndVal.append(testImages[i][1])
        yVal = testMoves

        xStartTrain = np.array(xStartTrain, np.float32)
        xEndTrain = np.array(xEndTrain, np.float32)
        xStartVal = np.array(xStartVal, np.float32)
        xEndVal = np.array(xEndVal, np.float32)


        return xStartTrain, xEndTrain, yTrain, xStartVal, xEndVal, yVal, inputShape


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class MoveClassifier:
    
    def __init__(self, pretrained = True, download_data = False):   
                
        if pretrained and download_data:
            self._download_weights()

        print("Loading Data...")
        self.xStartTrain, self.xEndTrain, self.yTrain, self.xStartVal, self.xEndVal, self.yVal, self.inputShape = load_data()

        print("Building Attack Model...")
        self.attackModel = self._build_model(regress=False)
        print("Building Forward Model...")
        self.forwardModel = self._build_model(regress=False)
        print("Building Backward Model...")
        self.backModel = self._build_model(regress=False)
        print("Building Left Model...")
        self.leftModel = self._build_model(regress=False)
        print("Building Right Model...")
        self.rightModel = self._build_model(regress=False)
        print("Building Jump Model...")
        self.jumpModel = self._build_model(regress=False)
        print("Building Sneak Model...")
        self.sneakModel = self._build_model(regress=False)
        print("Building Sprint Model...")
        self.sprintModel = self._build_model(regress=False)
        print("Building Use Model...")
        self.useModel = self._build_model(regress=False)
        print("Building Drop Model...")
        self.dropModel = self._build_model(regress=False)
        print("Building Inventory Model...")
        self.inventoryModel = self._build_model(regress=False)
        print("Building Hotbar Model...")
        self.hotbarModel = self._build_model(regress=False)
        print("Building Camera1 Model...")
        self.camera1Model = self._build_model(regress=True)
        print("Building Camera2 Model...")
        self.camera2Model = self._build_model(regress=True)
        print("Building ESC Model...")
        self.escModel = self._build_model(regress=False)

        print("Starting Training...")
        self._train()
        print("Training Complete!")
      
    def _create_cnn(self, width, height, depth, filters=(16, 32, 64), regress=False):
        #https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
        # initialize the input shape and channel dimension, assuming
        # TensorFlow/channels-last ordering
        inputShape = (int(height), int(width), int(depth))
        chanDim = -1
        # define the model input
        inputs = Input(shape=inputShape)
        # loop over the number of filters
        for (i, f) in enumerate(filters):
            # if this is the first CONV layer then set the input
            # appropriately
            if i == 0:
                x = inputs
            # CONV => RELU => BN => POOL
            x = Conv2D(f, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            # flatten the volume, then FC => RELU => BN => DROPOUT
            x = Flatten()(x) 
            x = Dense(16)(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Dropout(0.5)(x)


            x = Dense(4)(x)
            x = Activation("relu")(x)
            # check to see if the regression node should be added
            if regress:
                x = Dense(1, activation="linear")(x)
            # construct the CNN
            model = Model(inputs, x)
            # return the CNN
            return model
    
    def _build_model(self, regress=False):
        startCNN = self._create_cnn(self.inputShape[0], self.inputShape[1], self.inputShape[2], regress=regress)
        endCNN = self._create_cnn(self.inputShape[0], self.inputShape[1], self.inputShape[2], regress=regress)

        combinedInput = concatenate([startCNN.output, endCNN.output])

        x = Dense(4, activation="relu")(combinedInput)
        x = Dense(1, activation="linear")(x)

        model = Model(inputs=[startCNN.input, endCNN.input], outputs=x)

        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
        
        return model
    
    def _fit_generator(train_gen, val_gen, multiprocessing = False, workers = 1):
        if multiprocessing:
            train_gen = MultiprocessingGenerator(train_gen)
            val_gen = MultiprocessingGenerator(val_gen)
        return train_gen.fit_generator(val_gen, workers = workers)


    def _train(self, epochs=100, batch_size=8):
        
        print("Training Attack Model...")
        self.attackModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["attack"])), np.int64),
                       validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["attack"])), np.int64)),
                       epochs=epochs, 
                       batch_size=batch_size)
        self.save_model(self.attackModel, "attack.h5")
        
        print("Training Forward Model...")
        self.forwardModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["forward"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["forward"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.forwardModel, "forward.h5")

        print("Training Backward Model...")
        self.backModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["back"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["back"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.backModel, "back.h5")

        print("Training Left Model...")
        self.leftModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["left"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["left"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.leftModel, "left.h5")

        print("Training Right Model...")
        self.rightModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["right"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["right"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.rightModel, "right.h5")

        print("Training Jump Model...")
        self.jumpModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["jump"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["jump"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.jumpModel, "jump.h5")

        print("Training Sneak Model...")
        self.sneakModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["sneak"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["sneak"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.sneakModel, "sneak.h5")

        print("Training Sprint Model...")
        self.sprintModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["sprint"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["sprint"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.sprintModel, "sprint.h5")

        print("Training Use Model...")
        self.useModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["use"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["use"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.useModel, "use.h5")

        print("Training Drop Model...")
        self.dropModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["drop"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["drop"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.dropModel, "drop.h5")

        print("Training Inventory Model...")
        self.inventoryModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["inventory"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["inventory"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.inventoryModel, "inventory.h5")

        print("Training Hotbar Model...")
        self.hotbarModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["hotbar"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["hotbar"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.hotbarModel, "hotbar.h5")

        print("Training Camera1 Model...")
        self.camera1Model.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["camera1"])), np.float32),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["camera1"])), np.float32)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.camera1Model, "camera1.h5")

        print("Training Camera2 Model...")
        self.camera2Model.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["camera2"])), np.float32),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["camera2"])), np.float32)),
                        epochs=epochs,
                        batch_size=batch_size)
        self.save_model(self.camera2Model, "camera2.h5")

        print("Training ESC Model...")
        self.escModel.fit(x=[self.xStartTrain, self.xEndTrain], y=np.array(list(map(int,self.yTrain["ESC"])), np.int64),
                        validation_data=([self.xStartVal, self.xEndVal], np.array(list(map(int,self.yVal["ESC"])), np.int64)),
                        epochs=epochs,
                        batch_size=batch_size) 
        self.save_model(self.escModel, "ESC.h5")
    
    def save_model(self, model, name):
        model.save(name)

    def predict_move(self, startFrame, endFrame):
        move = {"attack": self.attackModel.predict([startFrame, endFrame]), 
                "forward": self.forwardModel.predict([startFrame, endFrame]), 
                "back": self.backModel.predict([startFrame, endFrame]),
                "left": self.leftModel.predict([startFrame, endFrame]),
                "right": self.rightModel.predict([startFrame, endFrame]),
                "jump": self.jumpModel.predict([startFrame, endFrame]),
                "sneak": self.sneakModel.predict([startFrame, endFrame]),
                "sprint": self.sprintModel.predict([startFrame, endFrame]),
                "use": self.useModel.predict([startFrame, endFrame]),
                "drop": self.dropModel.predict([startFrame, endFrame]),
                "inventory": self.inventoryModel.predict([startFrame, endFrame]),
                "hotbar": self.hotbarModel.predict([startFrame, endFrame]),
                "camera": [self.camera1Model.predict([startFrame, endFrame]), self.camera2Model.predict([startFrame, endFrame])],
                "ESC": self.escModel.predict([startFrame, endFrame])}
        self.model.predict([startFrame, endFrame])

        return move


    
    
