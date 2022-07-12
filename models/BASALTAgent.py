#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 19:12:02 2022

@author: chasebrown
"""

import InvClassifier as IC
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm
import time
import os
import pandas as pd
import json 

def paintInventory(img, inventory, cursorLocation):
    blockpath = "../assets/blockImages/"
    with open(blockpath + "colors.json") as json_file:
        colors = json.load(json_file)
    img1 = ImageDraw.Draw(img)
    font = ImageFont.truetype("../assets/fonts/abel-regular.ttf", 18)
    #Armor
    left = 240
    top = 105
    width = 16
    height = 16
    for item in range(0,4):
        img1.text((left, top),str(inventory["armor"][item]['quantity']),(0,0,0),font=font)
        top += 18
    
    left = 309
    top = 159
    img1.text((left, top),str(inventory["armor"][4]['quantity']),(0,0,0),font=font)
        
    
    #Inventory
    top = 181
    
    for row in range(0, 3):
        left = 240
        for column in range(0,9):
            img1.text((left, top),str(inventory["inventory"][row*9 + column]['quantity']),(0,0,0),font=font)

            left += 18
        top += 18
    
    #Item Bar
    top = 239
    left = 240
    
    for column in range(0,9):
        img1.text((left, top),str(inventory["item_bar"][column]['quantity']),(255,255,255),font=font)

        left += 18
        
    #Crafting
    top = 115
    
    for row in range(0, 2):
        left = 330
        for column in range(0,2):
            img1.text((left, top),str(inventory["crafting"][row*2 + column]['quantity']),(255,255,255),font=font)

            left += 18
        top += 18
    left = 386
    top = 125
    img1.text((left, top),str(inventory["crafting"][4]['quantity']),(255,255,255),font=font)

    
    
    shape = [(cursorLocation['x']-1, cursorLocation['y']-1), (cursorLocation['x'] + 1, cursorLocation['y'] + 1)]
    img1.rectangle(shape, fill= "blue", outline ="blue")
    
    img.save("./test.jpg")

def parseInvImage(img):
    
    img_data_array=[]
    
    #Armor
    left = 240
    top = 105
    width = 16
    height = 16
    for item in range(0,4):
        box = (left, top, left+width, top+height)
        image= np.array(img.crop(box))
        image = image.astype('float32')
        image /= 255  
        img_data_array.append(image)
        top += 18
    
    left = 309
    top = 159
    box = (left, top, left+width, top+height)
    image= np.array(img.crop(box))
    image = image.astype('float32')
    image /= 255  
    img_data_array.append(image)
    
    #Inventory
    top = 181
    
    for row in range(0, 3):
        left = 240
        for column in range(0,9):
            box = (left, top, left+width, top+height)
            image= np.array(img.crop(box))
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            left += 18
        top += 18
    
    #Item Bar
    top = 239
    left = 240
    
    for column in range(0,9):
        box = (left, top, left+width, top+height)
        image= np.array(img.crop(box))
        image = image.astype('float32')
        image /= 255  
        img_data_array.append(image)
        left += 18
        
    #Crafting
    top = 115
    
    for row in range(0, 2):
        left = 330
        for column in range(0,2):
            box = (left, top, left+width, top+height)
            image= np.array(img.crop(box))
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            left += 18
        top += 18
    left = 386
    top = 125
    box = (left, top, left+width, top+height)
    image= np.array(img.crop(box))
    image = image.astype('float32')
    image /= 255  
    img_data_array.append(image)

    xTestData = np.array(img_data_array, np.float32)

    n_samples_test = len(xTestData)
    forPredict = xTestData.reshape((n_samples_test, -1))
    
    return forPredict

class Agent:
    
    def __init__(self):  
       start = time.time()
       print("Initializing Agent...")
       #Important Variables
       self.model = self._build_model()
       self.invClassifier = IC.InvClassifier()
       
       self.inventory = {"armor": {0: {"item": "", "quantity": 0, "durability": 0},
                                  1: {"item": "", "quantity": 0, "durability": 0},
                                  2: {"item": "", "quantity": 0, "durability": 0},
                                  3: {"item": "", "quantity": 0, "durability": 0},
                                  4: {"item": "", "quantity": 0, "durability": 0}},
                        "item_bar": {0: {"item": "", "quantity": 0, "durability": 0},
                                                    1: {"item": "", "quantity": 0, "durability": 0},
                                                    2: {"item": "", "quantity": 0, "durability": 0},
                                                    3: {"item": "", "quantity": 0, "durability": 0},
                                                    4: {"item": "", "quantity": 0, "durability": 0},
                                                    5: {"item": "", "quantity": 0, "durability": 0},
                                                    6: {"item": "", "quantity": 0, "durability": 0},
                                                    7: {"item": "", "quantity": 0, "durability": 0},
                                                    8: {"item": "", "quantity": 0, "durability": 0}},
                        "inventory": {0: {"item": "", "quantity": 0, "durability": 0},
                                                    1: {"item": "", "quantity": 0, "durability": 0},
                                                    2: {"item": "", "quantity": 0, "durability": 0},
                                                    3: {"item": "", "quantity": 0, "durability": 0},
                                                    4: {"item": "", "quantity": 0, "durability": 0},
                                                    5: {"item": "", "quantity": 0, "durability": 0},
                                                    6: {"item": "", "quantity": 0, "durability": 0},
                                                    7: {"item": "", "quantity": 0, "durability": 0},
                                                    8: {"item": "", "quantity": 0, "durability": 0},
                                                    9: {"item": "", "quantity": 0, "durability": 0},
                                                    10: {"item": "", "quantity": 0, "durability": 0},
                                                    11: {"item": "", "quantity": 0, "durability": 0},
                                                    12: {"item": "", "quantity": 0, "durability": 0},
                                                    13: {"item": "", "quantity": 0, "durability": 0},
                                                    14: {"item": "", "quantity": 0, "durability": 0},
                                                    15: {"item": "", "quantity": 0, "durability": 0},
                                                    16: {"item": "", "quantity": 0, "durability": 0},
                                                    17: {"item": "", "quantity": 0, "durability": 0},
                                                    18: {"item": "", "quantity": 0, "durability": 0},
                                                    19: {"item": "", "quantity": 0, "durability": 0},
                                                    20: {"item": "", "quantity": 0, "durability": 0},
                                                    21: {"item": "", "quantity": 0, "durability": 0},
                                                    22: {"item": "", "quantity": 0, "durability": 0},
                                                    23: {"item": "", "quantity": 0, "durability": 0},
                                                    24: {"item": "", "quantity": 0, "durability": 0},
                                                    25: {"item": "", "quantity": 0, "durability": 0},
                                                    26: {"item": "", "quantity": 0, "durability": 0}},
                        "crafting": {0: {"item": "", "quantity": 0, "durability": 0},
                                                    1: {"item": "", "quantity": 0, "durability": 0},
                                                    2: {"item": "", "quantity": 0, "durability": 0},
                                                    3: {"item": "", "quantity": 0, "durability": 0},
                                                    4: {"item": "", "quantity": 0, "durability": 0}}}
       self.cursorLocation = {"x": 0, "y": 0}
       
       print("Agent Initialized. (Time Taken:", time.time()-start, ")")
       
    def _build_model(self):
        #Our Model
        pass
    
    def train(self):
        #Train Model
        pass
    
    def _observe_pov(self, pov):
        img = Image.fromarray(pov, 'RGB')
        invs = parseInvImage(img)
        
        items = self.invClassifier.predict_item(invs)
        quants = self.invClassifier.predict_quantity(invs)
        for slot in range(0, len(invs)):
            if slot < 5:
                self.inventory["armor"][slot]['item'] = items[slot]
                self.inventory["armor"][slot]['quantity'] = quants[slot]
            elif slot < 32:
                self.inventory["inventory"][slot-5]['item'] = items[slot]
                self.inventory["inventory"][slot-5]['quantity'] = quants[slot]
            elif slot < 41:
                self.inventory["item_bar"][slot-32]['item'] = items[slot]
                self.inventory["item_bar"][slot-32]['quantity'] = quants[slot]
            else:
                self.inventory["crafting"][slot-41]['item'] = items[slot]
                self.inventory["crafting"][slot-41]['quantity'] = quants[slot]
        self.cursorLocation = self.invClassifier.predict_cursor(cv2.cvtColor(pov, cv2.COLOR_RGB2BGR))
        
    
    def _show_mind(self, pov, console_print = False):
        
        if console_print:
            print("Current Inventory:")
            
            for section in self.inventory.keys():
                for cell in self.inventory[section].keys():
                    print(section, cell, "| Item:", self.inventory[section][cell]["item"], "| Quantity: ", self.inventory[section][cell]["quantity"])
                    
            print("Cursor Location: ", self.cursorLocation["x"], self.cursorLocation["y"])
        
        #img = Image.fromarray(pov, 'RGB')
        #paintInventory(img, self.inventory, self.cursorLocation)
        
        
        
    
    def act(self, obs):
        print("Starting Action...")
        start = time.time()
        augObs = self._observe_pov(obs['pov'])
        print("Agent Action Returned. (Time Taken:", time.time()-start, ")")

        return {'inventory': self.inventory, 'cursorLocation': self.cursorLocation}
        
        
    

    
    