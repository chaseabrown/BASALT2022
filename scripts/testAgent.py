#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:14:44 2022

@author: chasebrown
"""

import sys
sys.path.append('../models')
from BASALTAgent import Agent
from PIL import Image
import numpy as np

agent = Agent()

img = Image.open("/Users/chasebrown/Desktop/BASALT2022/BASALT2022-git/logs/Run Logs/D2022-07-06-T17-06-01/D2022-07-06-T17-07-40-718114.jpg")
img.show()

obs = {"pov": np.array(img)}

agent.act(obs)