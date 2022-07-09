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
from detecto import core, utils, visualize
import matplotlib.pyplot as plt
import cv2
import gdown

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
    trainPath = "../assets/datasets/Item Classifier Data/train/"

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
    trainPath = "../assets/datasets/Item Classifier Data/train/"

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
    
    def __init__(self, pretrained = True):   
                
        if pretrained:
            self._download_weights()
        self.item_model = self._build_inv_model()
        self.quant_model = self._build_inv_model()
        self.cursor_model = self._build_cursor_model()
        self.toNumDict = {}
        self.fromNumDict = {}
       
        self._train(pretrained)
       
       
    def _download_weights(self):
        url = 'https://drive.google.com/file/d/1OdH1n7362DGfeU6ZxBUwQVLcU6hRXJeV/view?usp=sharing'
        output = '../assets/datasets/Cursor Over Inventory/cursorFinderWeights.pth'
        gdown.download(url, output, quiet=False, fuzzy=True)
    
    
    def _build_inv_model(self):
        model = GaussianNB()
        
        return model
    
    def _build_cursor_model(self):
        model = GaussianNB()
        
        return model
    
    def _train(self, pretrained):
        x_train, y_train = load_quantity_data()
        self.quant_model.fit(x_train, y_train)
        
        x_train, y_train, fromNumDict, toNumDict = load_item_data()
        self.fromNumDict = fromNumDict
        self.toNumDict = toNumDict

        self.item_model.fit(x_train, y_train)
        
        if pretrained:
            self.cursor_model = core.Model.load('../assets/datasets/Cursor Over Inventory/cursorFinderWeights.pth', ['cursor'])
        else:
            dataset = core.Dataset('../assets/datasets/Cursor Over Inventory/train')
            loader = core.DataLoader(dataset, batch_size=2, shuffle=True)
            model = core.Model(['cursor'])
            
            valdataset = core.Dataset('../assets/datasets/Cursor Over Inventory/val')
            
            losses = model.fit(loader, valdataset, epochs=10, learning_rate=0.001, 
                               lr_step_size=5, verbose=True)
            
            self.cursor_model = model

    def predict_item(self, x):
        preds = []
        for pred in self.item_model.predict(x):
            preds.append(self.fromNumDict[pred])
        return preds
    
    def predict_quantity(self, x):
        pred = self.quant_model.predict(x)
        return pred
    
    def predict_cursor(self, x):
        predictions = self.cursor_model.predict_top(x)
        labels, boxes, scores = predictions
        try:
            return {'x': float(boxes[0][0]), 'y': float(boxes[0][1])}
        except:
            print(boxes)
            return {"x": 0, "y": 0}

    
    