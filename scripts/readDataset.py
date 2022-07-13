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
