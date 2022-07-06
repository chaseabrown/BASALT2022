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

template = """
<annotation>
	<folder>{train/test}</folder>
	<filename>{filename}</filename>
	<path>/Users/chasebrown/Desktop/BASALT2022/BASALT2022/assets/datasets/Cursor Over Inventory/{train/test}/{filename}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>640</width>
		<height>360</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>cursor</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{startHeight}</xmin>
			<ymin>{startWidth}</ymin>
			<xmax>{endHeight}</xmax>
			<ymax>{endWidth}</ymax>
		</bndbox>
	</object>
</annotation>
"""

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

path = "../assets/datasets/Cursor Over Inventory/"

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

path = "../assets/datasets/Cursor Over Inventory/"

newpath = path + "train/"
for file in os.listdir(newpath):    
    if "cursor" in file:
        curH = int(file.split("-")[-1].split('[')[1].split(']')[0])
        curW = int(file.split("-")[-1].split('[')[2].split(']')[0])
        with open(newpath + file.replace(".jpg", ".xml"), "w") as f:
            f.write(template.replace("{train/test}", "train").replace("{filename}", file).replace("{startHeight}", str(curH)).replace("{startWidth}", str(curW)).replace("{endHeight}", str(curH+16)).replace("{endWidth}", str(curW+10)))
        
path = "../assets/datasets/Cursor Over Inventory/"

newpath = path + "test/"
for file in os.listdir(newpath):    
    if "cursor" in file:
        curH = int(file.split("-")[-1].split('[')[1].split(']')[0])
        curW = int(file.split("-")[-1].split('[')[2].split(']')[0])
        with open(newpath + file.replace(".jpg", ".xml"), "w") as f:
            f.write(template.replace("{train/test}", "test").replace("{filename}", file).replace("{startHeight}", str(curH)).replace("{startWidth}", str(curW)).replace("{endHeight}", str(curH+16)).replace("{endWidth}", str(curW+10)))
        