#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:08:47 2022

@author: chasebrown
"""
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
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
import argparse
import locale
import json
import random
sys.stderr = stderr


class MoveClassifier:
    
    def __init__(self, inputShape, pretrained = True, download_data = False):   
        self.inputShape = inputShape
        
      
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

    def build_model_2Images(self, regress=False):
        startCNN = self._create_cnn(self.inputShape[0], self.inputShape[1], self.inputShape[2], regress=regress)
        endCNN = self._create_cnn(self.inputShape[0], self.inputShape[1], self.inputShape[2], regress=regress)

        combinedInput = concatenate([startCNN.output, endCNN.output])

        x = Dense(4, activation="relu")(combinedInput)
        if regress:
            x = Dense(1, activation="linear")(x)
        else:
            x = Dense(1, activation="relu")(x)

        model = Model(inputs=[startCNN.input, endCNN.input], outputs=x)

        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            
        model.summary()

        return model

    def build_model_1Image(self, regress=False):
        CNN = self._create_cnn(self.inputShape[0], self.inputShape[1], self.inputShape[2], regress=regress)

        x = Dense(4, activation="relu")(CNN.output)
        if regress:
            x = Dense(1, activation="linear")(x)
        else:
            x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=CNN.input, outputs=x)
        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            
        model.summary()

        return model

    def train(self, model,generator, val_generator, epochs):
        model.fit_generator(generator=generator,
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=6,
                epochs=epochs)
    
    def save_model(self, model, name):
        model

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


    
    
