# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:38:21 2020

@author: hardh
"""
import scipy.stats as stats

class TruncNorm():
    def __init__(self, mu, sigma=0.1, lower=0, upper=1):
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper 
        
    def draw(self):
        X = stats.truncnorm((self.lower - self.mu) / self.sigma,\
                            (self.upper - self.mu) / self.sigma, loc=self.mu, scale=self.sigma)
        sample = X.rvs(1)
        return sample
    
        