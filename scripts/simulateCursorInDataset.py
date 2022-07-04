#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:37:15 2022

@author: chasebrown
"""

from PIL import Image
import numpy as np
import os
from random import randrange

left = 320
top = 113
width = 10
height = 16


img = Image.open("../assets/datasets/Full Inventory Images/acacia_fence_gate-37-65.jpg")

box = (left, top, left+width, top+height)
cursor = img.crop(box)
arr = np.asarray(cursor)

for row in range(0, 16):
    for column in range(0, 10):
        if arr[row][column][0] > 40 and arr[row][column][0] < 220:
            arr[row][column][0] = 169
            arr[row][column][1] = 169
            arr[row][column][2] = 169

cursor = Image.fromarray(arr, 'RGB')

path = "../logs/Run Logs/Inventory/"

for file in os.listdir(path):    
    if "2.jpg" in file:
        for sampleNumber in range(0, 30):
            item = Image.open(path + file)
            itemArr = np.asarray(item)
            changed = False
            w, h = item.size
            startRow = randrange(h)
            startColumn = randrange(w)
            for row in range(0, 16):
                for column in range(0, 10):
                    try:
                        if not arr[row][column][0] == 169:
                            itemArr[startRow + row][startColumn + column][0] = arr[row][column][0]
                            itemArr[startRow + row][startColumn + column][1] = arr[row][column][1]
                            itemArr[startRow + row][startColumn + column][2] = arr[row][column][2]
                            changed=True
                    except:
                        pass
            if changed:
                itemWCursor = Image.fromarray(itemArr, 'RGB')
                itemWCursor.save(path + file.replace(".jpg", "") + "-(cursor[" + str(startRow) + "][" + str(startColumn) + "]).jpg")
                print(file.replace(".jpg", "") + "-(cursor[" + str(startRow) + "][" + str(startColumn) + "]).jpg")


