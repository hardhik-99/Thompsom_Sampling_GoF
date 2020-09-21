# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:46:21 2020

@author: hardh, aniq55
"""
import numpy as np
from constants import RATE_OFFLOAD, T_MAX, LOAD

class Server():
    def __init__(self, B_max, compute_cap):
        self.compute_cap = compute_cap
        self.B_max = B_max
        self.low = None
        self.high = None
        
    def epoch_specific(self, epoch_duration, primary_user_count):
        backlog_start = np.random.uniform(0, self.B_max)
        self.low = backlog_start
        self.high = backlog_start + (primary_user_count*RATE_OFFLOAD - self.compute_cap)
        
        if self.high < self.low:
            self.low, self.high = self.high, self.low
            
    def draw(self):
        sample_buffer_length = np.random.uniform(self.low, self.high)
        Tc = (sample_buffer_length + LOAD)/self.compute_cap
        if Tc<=T_MAX: 
            sample = (1, (T_MAX-Tc)/T_MAX)
        if Tc>T_MAX: 
            sample = (0, 1/Tc)
        return sample
    