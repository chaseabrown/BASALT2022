#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 21:15:25 2022

@author: chasebrown
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


def create_dataset_PIL(img_folder):
    
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


def showImage(img_folder, index):
    Image.open(img_folder + os.listdir(img_folder)[index]).show()
    

trainPath = "../assets/Useful Datasets/Item Classifier Data/train/"
testPath = "../assets/Useful Datasets/Item Classifier Data/test/"


xTrainData, yTrainData = create_dataset_PIL(trainPath)
xTestData, yTestData = create_dataset_PIL(testPath)


uniqueOutputs = []
for y in yTrainData:
    if not y in uniqueOutputs:
        uniqueOutputs.append(y)

toNumDict = {uniqueOutputs[i]: i for i in range(len(uniqueOutputs))}
fromNumDict = {i: uniqueOutputs[i] for i in range(len(uniqueOutputs))}


yNumTrain = [toNumDict[y] for y in yTrainData]


yNumTest = [toNumDict[y] for y in yTestData]


xTrainData = np.array(xTrainData, np.float32)
y_train = np.array(list(map(int,yNumTrain)), np.int64)

n_samples_train = len(xTrainData)
X_train = xTrainData.reshape((n_samples_train, -1))

xTestData = np.array(xTestData, np.float32)
y_test = np.array(list(map(int,yNumTest)), np.int64)

n_samples_test = len(xTestData)
X_test = xTestData.reshape((n_samples_test, -1))

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
GNB_classifier = GaussianNB()
GNB_classifier.fit(X_train, y_train)
predicted = GNB_classifier.predict(X_test)

for i in range(0, len(y_test)):
    if not y_test[i] == predicted[i]:
        print(fromNumDict[y_test[i]], fromNumDict[predicted[i]])


#print("\nClassification report for classifier %s:\n%s\n" % (GNB_classifier, metrics.classification_report(y_test, predicted)))
print("\nAccuracy of the Algorithm: ", GNB_classifier.score(X_test, y_test))
plt.show()

