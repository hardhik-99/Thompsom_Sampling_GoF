# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:11:04 2020

@author: hardh
"""
import numpy as np

n = 20
N = 20
#p_ref = 0.001 
p_ref = 1e-4 * 6    #TS-KS reference test statistic
num_arms = 3

T = 5000 #Time horizon
runs = 10
#change_points = [1000, 2000, 3000, 4000]
num_agents = 2
mean_ref = 0.1 #TS-CD reference test statistic

change_points = [int(np.random.exponential(1000))]  #Change-points follow a poisson process(lam=0.001)
while (change_points[-1] < T):
    change_points.append(int(change_points[-1] + np.random.exponential(1000)))
change_points.remove(change_points[-1])
print(change_points)