#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:08:47 2022

@author: chasebrown
"""

import os
import numpy as np
from PIL import Image
from sklearn.naive_bayes import GaussianNB

def create_dataset_quantity(img_folder):
    
    img_data_array=[]
    class_name=[]
    for file in os.listdir(img_folder):    
        if ".jpg" in file:
            image_path= os.path.join(img_folder, file)
            image= np.array(Image.open(image_path))
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            if file.split("-")[0] == "empty":
                class_name.append(1)
            else:
                class_name.append(int(file.split("-")[1].replace('.jpg', "")))
    return img_data_array , class_name

def load_quantity_data():
    trainPath = "../assets/Useful Datasets/Item Classifier Data/train/"

    xTrainData, yTrainData = create_dataset_quantity(trainPath)

    xTrainData = np.array(xTrainData, np.float32)
    y_train = np.array(list(map(int,yTrainData)), np.int64)

    n_samples_train = len(xTrainData)
    x_train = xTrainData.reshape((n_samples_train, -1))

    return x_train, y_train


def create_dataset_items(img_folder):
    
    img_data_array=[]
    class_name=[]
    for file in os.listdir(img_folder):    
        if ".jpg" in file:
            image_path= os.path.join(img_folder, file)
            image= np.array(Image.open(image_path))
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            class_name.append(file.split("-")[0])
    return img_data_array , class_name

def load_item_data():
    trainPath = "../assets/Useful Datasets/Item Classifier Data/train/"

    xTrainData, yTrainData = create_dataset_items(trainPath)

    uniqueOutputs = []
    for y in yTrainData:
        if not y in uniqueOutputs:
            uniqueOutputs.append(y)

    toNumDict = {uniqueOutputs[i]: i for i in range(len(uniqueOutputs))}
    fromNumDict = {i: uniqueOutputs[i] for i in range(len(uniqueOutputs))}


    yNumTrain = [toNumDict[y] for y in yTrainData]


    xTrainData = np.array(xTrainData, np.float32)
    y_train = np.array(list(map(int,yNumTrain)), np.int64)

    n_samples_train = len(xTrainData)
    x_train = xTrainData.reshape((n_samples_train, -1))

    return x_train, y_train, fromNumDict, toNumDict



class InvClassifier:
    
    def __init__(self):       
       self.item_model = self._build_model()
       self.quant_model = self._build_model()
       self.toNumDict = {}
       self.fromNumDict = {}
       
       self._train()
       
    def _build_model(self):
        model = GaussianNB()
        
        return model
    
    def _train(self):
        x_train, y_train = load_quantity_data()
        self.quant_model.fit(x_train, y_train)
        
        x_train, y_train, fromNumDict, toNumDict = load_item_data()
        self.fromNumDict = fromNumDict
        self.toNumDict = toNumDict

        self.item_model.fit(x_train, y_train)

    def predict_item(self, x):
        preds = []
        for pred in self.item_model.predict(x):
            preds.append(self.fromNumDict[pred])
        return preds
    
    def predict_quantity(self, x):
        pred = self.quant_model.predict(x)
        return pred

    
    