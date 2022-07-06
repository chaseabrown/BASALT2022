#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:53:45 2022

@author: chasebrown
"""



from detecto import core, utils, visualize



path = "../assets/datasets/Cursor Over Inventory/"

newpath = path + "train/"

dataset = core.Dataset(newpath)
model = core.Model(['cursor'])

model.fit(dataset)





