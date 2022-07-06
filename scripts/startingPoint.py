#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 21:14:22 2022

@author: chasebrown


Important Information:

Sample Action Space = OrderedDict([('ESC', 0), ('attack', 0), ('back', 0), ('camera', array([0., 0.], dtype=float32)), ('drop', 0), ('forward', 0),
                            ('hotbar.1', 0), ('hotbar.2', 0), ('hotbar.3', 0), ('hotbar.4', 0), ('hotbar.5', 0), ('hotbar.6', 0), ('hotbar.7', 0), ('hotbar.8', 0), ('hotbar.9', 0),
                            ('inventory', 0), ('jump', 0), ('left', 0), ('pickItem', 0), ('right', 0), ('sneak', 0), ('sprint', 0), ('swapHands', 0), ('use', 0)])

Sample Observation = {'pov': array([[[ 43,  57,  80],
                                     [ 43,  57,  80],
                                     [ 43,  57,  80],
                                     ...,
                                     [ 27,  15,   8],
                                     [ 27,  15,   8],
                                     [ 27,  15,   8]],
                             
                                    [[ 43,  58,  81],
                                     [ 43,  58,  81],
                                     [ 43,  58,  81],
                                     ...,
                                     [  0,   0,   0],
                                     [ 32,  17,   9],
                                     [ 32,  17,   9]],
                             
                                    [[ 44,  58,  82],
                                     [ 44,  58,  82],
                                     [ 44,  58,  82],
                                     ...,
                                     [170, 170, 170],
                                     [  0,   0,   0],
                                     [ 32,  17,   9]],
                             
                                    ...,
                             
                                    [[ 33,  32,  15],
                                     [ 33,  31,  14],
                                     [ 33,  32,  15],
                                     ...,
                                     [ 21,  11,   6],
                                     [ 21,  11,   6],
                                     [ 21,  11,   6]],
                             
                                    [[ 32,  31,  14],
                                     [ 32,  31,  14],
                                     [ 32,  31,  14],
                                     ...,
                                     [ 21,  11,   6],
                                     [ 21,  11,   6],
                                     [ 20,  11,   6]],
                             
                                    [[ 32,  31,  14],
                                     [ 32,  31,  14],
                                     [ 32,  31,  14],
                                     ...,
                                     [ 20,  10,   6],
                                     [ 20,  11,   6],
                                     [ 20,  11,   6]]], dtype=uint8)}
"""
import gym
import minerl
import logging
import coloredlogs
from PIL import Image
import datetime
import os

coloredlogs.install(logging.DEBUG)

def logRun(ac, obs, reward, done, info, startTimeSTR):
    datetimeSTR = datetime.datetime.now().strftime("D%Y-%m-%d-T%H-%M-%S-%f")
    log = open("./logs/Run Logs/" + startTimeSTR + "/" + datetimeSTR + ".txt", 'w+')
    for key in ac.keys():
        log.write(key + ": " + str(ac[key]) + "\n")
    log.write("\nReward: " + str(reward) + "\n")
    log.write("\nInfo: " + str(info) + "\n")
    log.write("\nDone: " + str(done) + "\n")
    
    log.close()
    
    img = Image.fromarray(obs['pov'], 'RGB')
    img.save("./logs/Run Logs/" + startTimeSTR + "/" + datetimeSTR + '.jpg')
    

startTimeSTR = datetimeSTR = datetime.datetime.now().strftime("D%Y-%m-%d-T%H-%M-%S")

path = "./logs/Run Logs/" + startTimeSTR
if not os.path.exists(path):
    os.makedirs(path)
    
env = gym.make("MineRLBasaltBuildVillageHouse-v0")
obs = env.reset()



done = False
counter = 0
while not done:
    counter += 1
    ac = env.action_space.noop()
    # Spin around to see what is around us
    ac["inventory"] = 1
    if counter < 10:
        ac["camera"] = [-10, 0]
    elif counter < 20:
        ac["camera"] = [0, -10]
    elif counter < 30:
        ac["camera"] = [10, 0]
    elif counter < 40:
        ac["camera"] = [0, 10]
        
    obs, reward, done, info = env.step(ac)
    env.render()
    if counter == 30:
        done = True
    logRun(ac, obs, reward, done, info, startTimeSTR)

env.close()
