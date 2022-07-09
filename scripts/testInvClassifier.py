#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:53:16 2022

@author: chasebrown
"""

import sys        
sys.path.append('../models')
import InvClassifier as IC
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
    testPath = "../assets/datasets/Item Classifier Data/test/"

    xTestData, yTestData = create_dataset_quantity(testPath)

    xTestData = np.array(xTestData, np.float32)
    y_test = np.array(list(map(int,yTestData)), np.int64)

    n_samples_test = len(xTestData)
    x_test = xTestData.reshape((n_samples_test, -1))

    return x_test, y_test


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
    testPath = "../assets/datasets/Item Classifier Data/test/"

    xTestData, yTestData = create_dataset_items(testPath)

    xTestData = np.array(xTestData, np.float32)
    y_test = yTestData

    n_samples_test = len(xTestData)
    x_test = xTestData.reshape((n_samples_test, -1))

    return x_test, y_test

# Item Classifier Test
invClassifier = IC.InvClassifier()

x_test, y_test = load_item_data()

predicted = invClassifier.predict_item(x_test)

numCorrect = 0
for i in range(0, len(y_test)):
    if not y_test[i] == predicted[i]:
        print(y_test[i], predicted[i])
    else:
        numCorrect += 1
    
print("\nAccuracy of Item Classifier: ", numCorrect/len(y_test))

# Quantity Classifier Test
x_test, y_test = load_quantity_data()

predicted = invClassifier.predict_quantity(x_test)

numCorrect = 0
for i in range(0, len(y_test)):
    if not y_test[i] == predicted[i]:
        print(y_test[i], predicted[i])
    else:
        numCorrect += 1
    
print("\nAccuracy of Quantity Classifier: ", numCorrect/len(y_test))





