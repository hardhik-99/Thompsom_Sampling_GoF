# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:41:25 2020

@author: hardh
"""
import numpy as np

class Binomial():
    def __init__(self, mu):
        self.mu = mu
        
    def draw(self):
        sample = np.random.binomial(1, self.mu, 1)
        return sample
    