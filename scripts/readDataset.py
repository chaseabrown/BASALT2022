#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 19:27:53 2022

@author: chasebrown
"""

#Read in the data
import json
import os
import cv2


task = 'MineRLBasaltFindCave-v0'
path = "../models/basalt-2022-behavioural-cloning-baseline-main/data/" + task + "/"

json_content = ""

games = {}
for file in os.listdir(path):
    with open(path + file, 'r') as json_file:
        if ".jsonl" in file:
            video = cv2.VideoCapture(path + file.replace(".jsonl", ".mp4"))
            listOfPOVs = []
            listOfMoves = []
            for line in json_file.readlines():
                try:
                    listOfMoves.append(json.loads(line))
                    success, image = video.read()
                    if success:
                        cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB, dst=image)
                        listOfPOVs.append(image)
                        success, image = video.read()
                except:
                    pass

            
                

            games.update({file.split(".")[0]: {"moves" : listOfMoves, "povs" : listOfPOVs}})
            print("Game added: " + file.split(".")[0])

for game in games.keys():
    print(len(games[game]["moves"])- len(games[game]["povs"])*2)


# pip install opencv-python

import cv2
import numpy as np

# video.mp4 is a video of 9 seconds
movesPath = "/Users/chasebrown/Desktop/BASALT2022/BASALT2022/models/basalt-2022-behavioural-cloning-baseline-main/data/MineRLBasaltFindCave-v0/Player43-2295ba9366f8-20220707-103050.jsonl"
filename = "/Users/chasebrown/Desktop/BASALT2022/BASALT2022/models/basalt-2022-behavioural-cloning-baseline-main/data/MineRLBasaltFindCave-v0/Player43-2295ba9366f8-20220707-103050.mp4"

cap = cv2.VideoCapture(filename)
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoFPS = int(cap.get(cv2.CAP_PROP_FPS))

print (f"frameCount: {frameCount}")
print (f"frameWidth: {frameWidth}")
print (f"frameHeight: {frameHeight}")
print (f"videoFPS: {videoFPS}")

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

print (f"DURATION: {frameCount/videoFPS}")

with open(movesPath, "r") as json_file:
    listOfMoves = []
    for line in json_file.readlines():
        try:
            listOfMoves.append(json.loads(line))
        except:
            print(line)

