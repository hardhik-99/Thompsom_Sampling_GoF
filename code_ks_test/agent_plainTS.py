# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 08:37:52 2020

@author: hardh
"""
from constants import *
import numpy as np
import random
from scipy.stats import ks_2samp
    
class ThomSamp_Plain():
    def __init__(self, n=n, N=N, p_ref=p_ref, num_arms=num_arms):
        self.n = n
        self.N = N
        self.p_ref = p_ref      #Reference statistic value for change detection
        self.num_arms = num_arms
        self.alpha = np.ones(self.num_arms) 
        self.beta = np.ones(self.num_arms)
        self.theta = np.ones(self.num_arms)
        self.count = np.zeros(self.num_arms)     #No. of times arm_i is pulled
        self.regret = np.zeros((runs, T+1))   #For Cumulative regret plot
        self.regret_norm = np.zeros((runs, T+1))    #For Normalized regret plot
        self.cache = []                           #Reward cache
        for i in np.arange(self.num_arms):
            self.cache.append([])
        return
                
    def reinitialize_param(self):   #reset parameters after change detection
        self.alpha = np.ones(self.num_arms)
        self.beta = np.ones(self.num_arms)
        self.count = np.zeros(self.num_arms)
        return
        
    def update_count(self, i):  #Update count for arm_i
        self.count[i] = self.count[i] + 1
    
    def select_arm(self):
        for i in np.arange(self.num_arms):
            self.theta[i] = np.random.beta(self.alpha[i], self.beta[i])
            
        max_i = []
        max_arm_val = np.max(self.theta)
        for i in np.arange(self.num_arms):
            if (self.theta[i] == max_arm_val):
                max_i.append(i)
            
        arm_pull = random.choice(max_i)
        return arm_pull
        
    def update_param(self, x, i):
        if (x==1):
            self.alpha[i] = self.alpha[i] + 1
            self.beta[i] = self.beta[i] + 0
        else:
            self.alpha[i] = self.alpha[i] + 0
            self.beta[i] = self.beta[i] + 1
        return
            
    def store_reward(self, r, i):
        self.cache[i].append(r)
        if (len(self.cache[i]) > self.n+self.N+1):
            self.cache[i].remove(self.cache[i][0])
        return
    
    def change_detection(self, i):      #Active change point detection
        if (self.count[i] > (self.N + self.n)):
            test = np.ravel(self.cache[i][-self.n:])
            estimate = np.ravel(self.cache[i][(-self.n-self.N):(-self.n)])
            stat, p = ks_2samp(estimate, test)
            if(p < self.p_ref): #Always return False (Plain TS)
                return False
            else:
                return False
        else:
            return False
    
    def update_regret(self, run, t, max_mu, selected_mu):
        self.regret[run, t+1] = self.regret[run, t] + max_mu - selected_mu
        self.regret_norm[run, t+1] = self.regret[run, t+1] * (1/(t+1))
        
        

        