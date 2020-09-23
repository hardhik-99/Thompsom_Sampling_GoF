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
        self.high = backlog_start + (primary_user_count*RATE_OFFLOAD - self.compute_cap)*epoch_duration
        
        if self.high < self.low:
            self.low, self.high = self.high, self.low
            
        if self.low < 0:
            self.low = 0
            
        if self.high > self.B_max:
            self.high = self.B_max
            
        #print(primary_user_count*RATE_OFFLOAD, self.compute_cap)  
                
    def draw(self):
        sample_buffer_length = np.random.uniform(self.low, self.high)
        #print(self.low, self.high)
        Tc = (sample_buffer_length + LOAD)/self.compute_cap
        """
        print("Buffer Length:",sample_buffer_length)
        print("Load:",LOAD)
        print("Compute Cap:",self.compute_cap)
        print("Tc: ",Tc)
        """
        if Tc<=T_MAX: 
            sample = (1, T_MAX-Tc)
        if Tc>T_MAX: 
            sample = (0, T_MAX-Tc)
        
        return sample
    