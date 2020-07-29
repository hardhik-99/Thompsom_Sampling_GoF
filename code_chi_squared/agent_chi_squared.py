# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 08:37:52 2020

@author: hardh
"""
from constants import *
import numpy as np
import random
from scipy.stats import chisquare

class ThomSamp_ChiSquared():
    def __init__(self, n=n, N=N, p_ref=p_ref, num_arms=num_arms):
        self.n = n
        self.N = N
        self.p_cs_ref = p_cs_ref
        self.num_arms = num_arms
        self.alpha = np.ones(self.num_arms)
        self.beta = np.ones(self.num_arms)
        self.theta = np.ones(self.num_arms)
        self.count = np.zeros(self.num_arms)
        self.regret = np.zeros((runs, T+1))
        self.regret_norm = np.zeros((runs, T+1))
        self.cache = []
        for i in np.arange(self.num_arms):
            self.cache.append([])
        self.bin_cache = []
        for i_bin in np.arange(self.num_arms):
            self.bin_cache.append([])
        return
        
    def reinitialize_param(self):
        self.alpha = np.ones(self.num_arms)
        self.beta = np.ones(self.num_arms)
        self.count = np.zeros(self.num_arms)
        return
        
    def update_count(self, i):
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
        if (len(self.cache[i]) > self.n+self.N):
            self.cache[i].remove(self.cache[i][0])
        return
    
    def store_bin_reward(self, r, i):
        self.bin_cache[i].append(r)
        if (len(self.bin_cache[i]) > self.n):
            self.bin_cache[i].remove(self.bin_cache[i][0])
        return
    
    def change_detection(self, i):
        if (self.count[i] > (self.N + self.n)):
            
            p_exp = np.mean(np.ravel(self.cache[i][(-self.n-self.N):(-self.n)]))
            
            exp_1 = self.n * p_exp
            exp_0 = self.n * (1-p_exp)
            
            f_exp = np.array([exp_0, exp_1])
            
            test = np.ravel(self.bin_cache[i][-self.n:])
            
            obs_1 = np.count_nonzero(test == 1)
            obs_0 = self.n - obs_1
            
            f_obs = np.array([obs_0, obs_1])
            
            stat, p = chisquare(f_obs, f_exp)
            
            if(p < self.p_cs_ref):
                return True
            else:
                return False
        else:
            return False
        
    def update_regret(self, run, t, max_mu, selected_mu):
        self.regret[run, t+1] = self.regret[run, t] + max_mu - selected_mu
        self.regret_norm[run, t+1] = self.regret[run, t+1] * (1/(t+1))
        
        
        
            
        
        