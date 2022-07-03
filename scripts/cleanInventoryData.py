#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:39:44 2022

@author: chasebrown
"""

from PIL import Image
import os
import numpy
import pandas as pd

def parseInvImage(img):
    #Armor
    left = 240
    top = 105
    width = 16
    height = 16
    armor = {}
    for item in range(0,4):
        box = (left, top, left+width, top+height)
        armor.update({item: img.crop(box)})
        top += 18
    
    left = 309
    top = 159
    box = (left, top, left+width, top+height)
    armor.update({4: img.crop(box)})
    
    #Inventory
    top = 181
    inventory = {}
    
    for row in range(0, 3):
        left = 240
        for column in range(0,9):
            box = (left, top, left+width, top+height)
            inventory.update({row*9 + column: img.crop(box)})
            left += 18
        top += 18
    
    #Item Bar
    top = 239
    left = 240
    itemBar = {}
    
    for column in range(0,9):
        box = (left, top, left+width, top+height)
        itemBar.update({column: img.crop(box)})
        left += 18
    
    #Crafting
    top = 115
    crafting = {}
    
    for row in range(0, 2):
        left = 330
        for column in range(0,2):
            box = (left, top, left+width, top+height)
            crafting.update({row*2 + column: img.crop(box)})
            left += 18
        top += 18
    left = 386
    top = 125
    box = (left, top, left+width, top+height)
    crafting.update({4: img.crop(box)})
    
    return {"Armor": armor, "Inventory": inventory, "Item Bar": itemBar, "Crafting": crafting}

inputPath = "../assets/Useful Datasets/Full Inventory Images/"
trainPath = "../assets/Useful Datasets/Item Classifier Data/train/"
testPath = "../assets/Useful Datasets/Item Classifier Data/test/"

#Training Data

files = os.listdir(inputPath)
pictures = []
for file in files:
    if '.jpg' in file and not "random" in file:
        filename = file
        file = file.split('.')[0]
        splitname = file.split('-')
        block = splitname[0]
        start = splitname[1]
        end = splitname[2]
        pictures.append({"block": block, "start": start, "end": end, "image": parseInvImage(Image.open(inputPath + filename))})

listOfItems = []    
for picture in pictures:
    if 'air' == picture['block'][:3]:
        picture['block'] = 'empty'
        for i in range(0,9):
            listOfItems.append({"block": picture["block"], "quantity": i+1, "image": picture['image']['Item Bar'][i]})
        for row in range(0,3):
            for column in range(0,9):
                listOfItems.append({"block": picture["block"], "quantity": row*9 + column + 10, "image": picture['image']['Inventory'][row*9 + column]})
    else:
        if picture['start'] == "1":
            for i in range(0,9):
                listOfItems.append({"block": picture["block"], "quantity": i+1, "image": picture['image']['Item Bar'][i]})
            for row in range(0,3):
                for column in range(0,9):
                    listOfItems.append({"block": picture["block"], "quantity": row*9 + column + 10, "image": picture['image']['Inventory'][row*9 + column]})
        else:
            for i in range(0,9):
                listOfItems.append({"block": picture["block"], "quantity": i+37, "image": picture['image']['Item Bar'][i]})
            for row in range(0,3):
                for column in range(0,9):
                    if row*9 + column >= 19:
                        break
                    listOfItems.append({"block": picture["block"], "quantity": row*9 + column + 46, "image": picture['image']['Inventory'][row*9 + column]})

for item in listOfItems:
    item['image'].save(trainPath + item['block'] + "-" + str(item['quantity']) + ".jpg")



#Testing Data

files = os.listdir(inputPath)
pictures = []

for file in files:
    if '].jpg' in file:
        pictures.append({"image": Image.open(inputPath + file), "key": pd.read_csv(inputPath + file.replace(".jpg", ".csv"))})

for picture in pictures:
    items = parseInvImage(picture['image'])
    for index, row in picture['key'].iterrows():
        if row['block'] == "empty":
            if row['location'] < 9:
                items["Item Bar"][row['location']].save(testPath + row['block'] + "-" + str(row['location']) + ".jpg")
            else:
                items["Inventory"][row['location'] - 9].save(testPath + row['block'] + "-" + str(row['location']) + ".jpg")
        else:
            if row['location'] < 9:
                items["Item Bar"][row['location']].save(testPath + row['block'] + "-" + str(row['quantity']) + ".jpg")
            else:
                items["Inventory"][row['location'] - 9].save(testPath + row['block'] + "-" + str(row['quantity']) + ".jpg")
        

