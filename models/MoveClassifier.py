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
import sys
import pandas as pd
import random
sys.path.append('../helpers')
from Generators import Generator

def gatherData(dataPathList, framesViewed):
    folders = []
    images = []
    labels = {"attack": [], 
               "forward": [], 
               "back": [], 
               "left": [], 
               "right": [], 
               "jump": [], 
               "sneak": [], 
               "sprint": [], 
               "use": [], 
               "drop": [], 
               "inventory": [], 
               "hotbar": [], 
               "camera1": [], 
               "camera2": []}
    
    for dataPath in dataPathList:
        for folder in os.listdir(dataPath):
            folders.append(dataPath + folder)
            if not ".DS_Store" in folder:
                newMoves = pd.read_csv(dataPath + folder + "/moves.csv")
                for index, move in newMoves.iterrows():
                    framesToInclude = []
                    for i in range(0, framesViewed):
                        framesToInclude.append(dataPath + folder + "/" + str(int(move["startImage"] + i)) + ".jpg")
                    images.append(framesToInclude)
                    labels["attack"].append(move["attack"])
                    labels["forward"].append(move["forward"])
                    labels["back"].append(move["back"])
                    labels["left"].append(move["left"])
                    labels["right"].append(move["right"])
                    labels["jump"].append(move["jump"])
                    labels["sneak"].append(move["sneak"])
                    labels["sprint"].append(move["sprint"])
                    labels["use"].append(move["use"])
                    labels["drop"].append(move["drop"])
                    labels["inventory"].append(move["inventory"])
                    labels["hotbar"].append(move["hotbar"])
                    labels["camera1"].append(move["camera1"])
                    labels["camera2"].append(move["camera2"])
                    
    return images, labels

class MoveClassifier:
    
    def __init__(self, pretrained = True, download_data = False):   
                
        print("Loading Data...")
        DATAPATHS = ["../assets/datasets/Move Classifier Data/MineRLBasaltFindCave-v0/", 
                            "../assets/datasets/Move Classifier Data/MineRLBasaltBuildVillageHouse-v0/", 
                            "../assets/datasets/Move Classifier Data/MineRLBasaltCreateVillageAnimalPen-v0/", 
                            "../assets/datasets/Move Classifier Data/MineRLBasaltMakeWaterfall-v0/"]
        images, labels = gatherData(DATAPATHS, 2)
        
        keys = ["attack", "forward", "back", "left", "right", "jump", "sneak", "sprint", "use", "drop", "inventory", "hotbar", "camera1", "camera2"]



        temp = list(zip(images, labels["attack"]))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        # res1 and res2 come out as tuples, and so must be converted to lists.
        images, labels = list(res1), list(res2)

        X_train = images[:int(len(images) * 0.8)]
        Y_train = labels[:int(len(labels) * 0.8)]
        X_val = images[int(len(images) * 0.8):]
        Y_val = labels[int(len(labels) * 0.8):]

        self.generator = Generator(X_train, Y_train, batch_size=16)
        self.val_generator = Generator(X_val, Y_val, batch_size=16)

        self.inputShape = (640, 360, 3)

        print("Building Attack Model...")
        self.attackModel = self._build_model(regress=False)
        """
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
"""
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

    def _train(self, epochs=5):
        
        print("Training Attack Model...")
        self.attackModel.fit_generator(generator=self.generator,
                    validation_data=self.val_generator,
                    epochs=epochs)
        self.save_model(self.attackModel, "attack.h5")
        """
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
    """
    
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


    
    
