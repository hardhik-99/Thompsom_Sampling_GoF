# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:58:11 2020

@author: hardh
"""
import sys
from agent import ThomSamp_GoF
from agent_cd import ThomSamp_MeanDiff
from arm_reward_truncnorm import TruncNorm
from constants import *
import numpy as np
import random
import matplotlib.pyplot as plt

A = [] #Agents list
A.append(ThomSamp_MeanDiff()) #Agent 0 (TS-CD)
A.append(ThomSamp_GoF())    #Agent 1 (TS-KS)

for run in np.arange(runs):
    
    change_points = [int(np.random.exponential(1000))]  #Change-points follow a poisson process(lam=0.001)
    while (change_points[-1] < T):
        change_points.append(int(change_points[-1] + np.random.exponential(1000)))
    change_points.remove(change_points[-1])
    print(change_points)
            
    arm = []
    mu = np.random.uniform(0, 1, num_arms)
    #max_mu = np.max(mu)
    #print(mu)
    for i in np.arange(num_arms):
        arm.append(TruncNorm(mu[i]))

    for t in np.arange(T):
        if t in change_points:
            arm = []
            mu = np.random.uniform(0, 1, num_arms)
            #max_mu = np.max(mu)
            for i in np.arange(num_arms):
                arm.append(TruncNorm(mu[i]))
            #print(mu)
        
        for i in range(num_agents):
            reward_all_arms = [arm[j].draw() for j in range(num_arms)]
            max_reward_t = max(reward_all_arms)
            k = A[i].select_arm()
            #r = arm[k].draw()
            r = reward_all_arms[k]
            A[i].store_reward(r, k)
            b = np.random.binomial(1, r, 1)
            A[i].update_param(b, k)
            A[i].update_count(k)
            A[i].update_regret(run, t, max_reward_t, r)
        
            best_k = np.argmax(A[i].alpha/(A[i].alpha + A[i].beta)) #Current best arm
        
            if(A[i].change_detection(best_k)):
                A[i].reinitialize_param()
                #print("Change time pred. =", i, "---->", t)
                
    #if ((run%10)==0):
        #sys.stdout.write('.'); sys.stdout.flush();
    for i in range(num_agents):
        A[i].reinitialize_param() #Reinitialize before next run
        print("Run:",run,"TS-KS-Cumulative:", i, "----->",A[i].regret[run, t+1]) #Print current episode's cumulative regret
    print("\n")
        
#********************************************************************#
regret_ks_avg = []
regret_ks_std = []
 
#Plot cumulative regret graph
for i in range(num_agents):
    regret_ks_avg.append(np.mean(A[i].regret, axis=0))
    regret_ks_std.append(np.std(A[i].regret, axis=0))
    print("TS-KS-Cumulative", i, ":",regret_ks_avg[i][t])

time_arr = np.arange(T+1)

plt.figure(figsize=(15,7.5), dpi= 80)
plt.plot(time_arr, regret_ks_avg[0],'b',markersize = 0.2, label="TS-CD")
plt.fill_between(time_arr,\
        regret_ks_avg[0]-regret_ks_std[0], \
        regret_ks_avg[0]+regret_ks_std[0], \
        color='b', alpha=0.2)
plt.plot(time_arr, regret_ks_avg[1],'r',markersize = 0.2, label="TS-KS")
plt.fill_between(time_arr,\
        regret_ks_avg[1]-regret_ks_std[1], \
        regret_ks_avg[1]+regret_ks_std[1], \
        color='r', alpha=0.2)
plt.xlabel("Time Step----->")
plt.ylabel("Cumulative regret----->")
plt.title("Non-Stationary TS")
plt.legend(loc="upper left")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.grid()
plt.show()

#Plot Normalized regret graph
regret_norm_ks_avg = []
regret_norm_ks_std = []

for i in range(num_agents):
    regret_norm_ks_avg.append(np.mean(A[i].regret_norm, axis=0))
    regret_norm_ks_std.append(np.std(A[i].regret_norm, axis=0))
    print("TS-KS-Normalized", i, ":",regret_norm_ks_avg[i][t])

time_arr = np.arange(T+1)

plt.figure(figsize=(15,7.5), dpi= 80)
plt.plot(time_arr, regret_norm_ks_avg[0],'b',markersize = 0.2, label="TS-CD")
plt.fill_between(time_arr,\
    regret_norm_ks_avg[0]-regret_norm_ks_std[0], \
    regret_norm_ks_avg[0]+regret_norm_ks_std[0], \
    color='b', alpha=0.2)
plt.plot(time_arr, regret_norm_ks_avg[1],'r',markersize = 0.2, label="TS-KS")
plt.fill_between(time_arr,\
    regret_norm_ks_avg[1]-regret_norm_ks_std[1], \
    regret_norm_ks_avg[1]+regret_norm_ks_std[1], \
    color='r', alpha=0.2)
plt.xlabel("Time Step----->")
plt.ylabel("Normalized regret----->")
plt.title("Non-Stationary TS")
plt.legend(loc="upper left")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.grid()
plt.show()
    
