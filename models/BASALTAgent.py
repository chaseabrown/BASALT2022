#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 19:12:02 2022

@author: chasebrown
"""

import InvClassifier as IC

def augmentObs(pov, inventory):
    pass

class Agent:
    
    def __init__(self):       
       #Important Variables
       self.model = self._build_model()
       self.invClassifier = IC.InvClassifier()
       
       self.inventory = {}
       
    def _build_model(self):
        #Our Model
        pass
    
    def train(self):
        #Train Model
        pass
    
    def act(self, obs):
        augObs = augmentObs(obs['pov'])
        action = self.model.predict(augObs)
        
    

    
    