# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:11:04 2020

@author: hardh, aniq55
"""
import numpy as np

n = 30
N = 30
#p_ref = 0.001 
p_ref = 1e-4 * 6    #TS-KS reference test statistic
num_arms = 3

T = 5000 #Time horizon
runs = 100
#change_points = [1000, 2000, 3000, 4000]
num_agents = 2
mean_ref = 0.1 #TS-CD reference test statistic
"""
epoch_durations = [int(np.random.exponential(1000))]  #Change-points follow a poisson process(lam=0.001)

change_points = epoch_durations
while (change_points[-1] < T):
    change_points.append(int(change_points[-1] + np.random.exponential(1000)))
change_points.remove(change_points[-1])
print(len(change_points))
"""


T_MAX = 0.1 # need to see the values of T_c to fix this
RATE_OFFLOAD = 1024.0*10**3 # 8 kbps
LOAD = 20.0*10**6 # 20 MB * 1 cycle/byte


B_max_max = 1024*10**6 # 1 GB
B_max_min = 512.0*10**6 # 0.5 GB
#print(B_max_max)
compute_cap_max = 4.0*10**9 # 4 GHz
compute_cap_min = 2.0*10**9 # 2 GHz


TOTAL_USERS = 1000 # vary from 10 to 100 in steps of 100
p_vals_uniform = (1.0/num_arms)*np.ones(num_arms)
