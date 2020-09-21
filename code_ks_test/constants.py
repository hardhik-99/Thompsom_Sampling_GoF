# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:11:04 2020

@author: hardh, aniq55
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
"""
epoch_durations = [int(np.random.exponential(1000))]  #Change-points follow a poisson process(lam=0.001)

change_points = epoch_durations
while (change_points[-1] < T):
    change_points.append(int(change_points[-1] + np.random.exponential(1000)))
change_points.remove(change_points[-1])
print(len(change_points))
"""
epoch_durations = np.random.exponential(1000, 10).astype(int)
change_points = np.cumsum(epoch_durations)
print(change_points)

T_MAX = 10
RATE_OFFLOAD = 10
LOAD = 10


B_max_max = 100
B_max_min = 10

compute_cap_max = 100
compute_cap_min = 100


TOTAL_USERS = 100

p_vals_uniform = (1.0/num_arms)*np.ones(num_arms)
