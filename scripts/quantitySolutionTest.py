#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 19:27:53 2022

@author: chasebrown
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
            if file.split("-")[0] == "empty":
                class_name.append(1)
            else:
                class_name.append(int(file.split("-")[1].replace('.jpg', "")))
    return img_data_array , class_name


def showImage(img_folder, index):
    Image.open(img_folder + os.listdir(img_folder)[index]).show()
    

trainPath = "../assets/Useful Datasets/Item Classifier Data/train/"
testPath = "../assets/Useful Datasets/Item Classifier Data/test/"


xTrainData, yTrainData = create_dataset_PIL(trainPath)
xTestData, yTestData = create_dataset_PIL(testPath)

xTrainData = np.array(xTrainData, np.float32)
y_train = np.array(list(map(int,yTrainData)), np.int64)

n_samples_train = len(xTrainData)
X_train = xTrainData.reshape((n_samples_train, -1))

xTestData = np.array(xTestData, np.float32)
y_test = np.array(list(map(int,yTestData)), np.int64)

n_samples_test = len(xTestData)
X_test = xTestData.reshape((n_samples_test, -1))


# Gaussian Naive Bayes
GNB_classifier = GaussianNB()
GNB_classifier.fit(X_train, y_train)
predicted = GNB_classifier.predict(X_test)

for i in range(0, len(y_test)):
    if not y_test[i] == predicted[i]:
        print(y_test[i], predicted[i])

print("\nClassification report for classifier %s:\n%s\n" % (GNB_classifier, metrics.classification_report(y_test, predicted)))
print("\nAccuracy of the Algorithm: ", GNB_classifier.score(X_test, y_test))
plt.show()




"""
# Support Vector Machines (SVM)
from sklearn import svm
svm_classifier = svm.SVC(gamma=0.001)
svm_classifier.fit(X_train, y_train)
predicted = svm_classifier.predict(X_test)
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(xData, yData))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
    
images_and_predictions = list(zip(xData[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)
print("\nClassification report for classifier %s:\n%s\n" % (svm_classifier, metrics.classification_report(y_test, predicted)))
print("\nAccuracy of the Algorithm: ", svm_classifier.score(X_test, y_test))
plt.show()



# Decision Trees
from sklearn import tree
dt_classifier = tree.DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
predicted = dt_classifier.predict(X_test)
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(xData, yData))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
    
images_and_predictions = list(zip(xData[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)
print("\nClassification report for classifier %s:\n%s\n" % (dt_classifier, metrics.classification_report(y_test, predicted)))

print("\nAccuracy of the Algorithm: ", dt_classifier.score(X_test, y_test))
plt.show()


# Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(max_depth=2, random_state=0)
RF_classifier.fit(X_train, y_train)
predicted = RF_classifier.predict(X_test)
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(xData, yData))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
    
images_and_predictions = list(zip(xData[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)
print("\nClassification report for classifier %s:\n%s\n" % (RF_classifier, metrics.classification_report(y_test, predicted)))

print("\nAccuracy of the Algorithm: ", RF_classifier.score(X_test, y_test))
plt.show()

# K Nearest Neighbours (KNN)
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
KNN_classifier.fit(X_train, y_train)
predicted = KNN_classifier.predict(X_test)
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(xData, yData))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
    
images_and_predictions = list(zip(xData[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)
print("\nClassification report for classifier %s:\n%s\n" % (KNN_classifier, metrics.classification_report(y_test, predicted)))
print("\nAccuracy of the Algorithm: ", KNN_classifier.score(X_test, y_test))
plt.show()


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
sgd_classifier.fit(X_train, y_train)
predicted = sgd_classifier.predict(X_test)
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(xData, yData))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
    
images_and_predictions = list(zip(xData[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)
print("\nClassification report for classifier %s:\n%s\n" % (sgd_classifier, metrics.classification_report(y_test, predicted)))
print("\nAccuracy of the Algorithm: ", sgd_classifier.score(X_test, y_test))
plt.show()

"""