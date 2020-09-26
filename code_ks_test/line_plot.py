# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:48:39 2020

@author: hardh
"""
from agent import ThomSamp_GoF
from agent_cd import ThomSamp_MeanDiff
from agent_plainTS import ThomSamp_Plain
from agent_DiscountedTS import ThomSamp_Discounted
from arm_reward_truncnorm import TruncNorm
from arm_reward_server import Server
from constants import *
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24}) 
import pandas as pd 

num_mean_epochs = 10

A = [] #Agents list
A.append(ThomSamp_MeanDiff()) #Agent 0 (TS-CD)
A.append(ThomSamp_GoF())    #Agent 1 (TS-KS)
A.append(ThomSamp_Plain())  #Agent 2 (TS)
A.append(ThomSamp_Discounted())  #Agent 3 (Discounted TS)

label = ["TS-CD", "TS-KS", "TS", "Discounted TS"]

sat_cum_regret = np.zeros((num_agents, num_mean_epochs))
sat_norm_regret = np.zeros((num_agents, num_mean_epochs))

epoch_dur = np.arange(50, 550, 50)

for mean_epoch_dur in epoch_dur:
    
    for run in np.arange(runs):
    
        epoch_durations = np.random.exponential(mean_epoch_dur, 500).astype(int)
        change_points = np.cumsum(epoch_durations)
        #print(change_points)

        arm = []
        for i in np.arange(num_arms):
            B_max_this = np.random.uniform(B_max_min, B_max_max)
            compute_cap_this = np.random.uniform(compute_cap_min, compute_cap_max)
            arm.append(Server(B_max_this, compute_cap_this)) #Add reward distribution of the arm
            
            epoch_count = 0
            user_split = np.random.multinomial(TOTAL_USERS, p_vals_uniform)
            
        for i in np.arange(num_arms):
            arm[i].epoch_specific(epoch_durations[epoch_count], user_split[i])

            epoch_count = epoch_count + 1

        for t in np.arange(T):
            if t in change_points:
                # re-defining the arms at change points
                user_split = np.random.multinomial(TOTAL_USERS, p_vals_uniform)
                for i in np.arange(num_arms):
                    arm[i].epoch_specific(epoch_durations[0], user_split[i])
                    epoch_count = epoch_count + 1
                    
            for i in range(num_agents):
                reward_all_arms = [arm[j].draw() for j in range(num_arms)]
                k = A[i].select_arm()
                r = reward_all_arms[k][1]
                A[i].store_reward(r, k)
                #b = np.random.binomial(1, r, 1)
                b = reward_all_arms[k][0]
                A[i].update_param(b, k)
                A[i].update_count(k)
                
                A[i].update_regret(run, t, max([x[1] for x in reward_all_arms]), r)
                
                best_k = np.argmax(A[i].alpha/(A[i].alpha + A[i].beta)) #Current best arm
                
                if(A[i].change_detection(best_k)):
                    A[i].reinitialize_param()
                    #print("Change time pred. =", i, "---->", t)

        for i in range(num_agents):
            A[i].reinitialize_param() #Reinitialize before next run
            print("Run:",run,"TS-KS-Cumulative:", i, "----->",A[i].regret[run, t+1]) #Print current episode's cumulative regret
        print("\n")
    
    regret_ks_avg = []
    regret_ks_std = []
    regret_norm_ks_avg = []
    regret_norm_ks_std = []
        
    for i in range(num_agents):
        regret_ks_avg.append(np.mean(A[i].regret, axis=0))
        regret_ks_std.append(np.std(A[i].regret, axis=0))
        print("TS-KS-Cumulative", i, ":",regret_ks_avg[i][t])
        
    for i in range(num_agents):
        regret_norm_ks_avg.append(np.mean(A[i].regret_norm, axis=0))
        regret_norm_ks_std.append(np.std(A[i].regret_norm, axis=0))
        print("TS-KS-Normalized", i, ":",regret_norm_ks_avg[i][t])
    
    for i in range(num_agents):
        sat_cum_regret[i][int(mean_epoch_dur/50)-1] = regret_ks_avg[i][t]
        sat_norm_regret[i][int(mean_epoch_dur/50)-1] = regret_norm_ks_avg[i][t]
#********************************************************************#
#pd.DataFrame(sat_cum_regret).to_csv(r"C:\Users\hardh\Desktop\Reinforcement_Learning\sat_cum_regret.csv")
#pd.DataFrame(sat_norm_regret).to_csv(r"C:\Users\hardh\Desktop\Reinforcement_Learning\sat_norm_regret.csv")


plt.figure(figsize=(15,7.5), dpi=600)
for i in range(num_agents): 
    plt.plot(epoch_dur, sat_cum_regret[i], label=label[i])
plt.xlabel("Mean epoch duration")
plt.ylabel("Cumulative regret")
#plt.title("Non-Stationary TS")
plt.legend(loc="best")
plt.ylim(ymin=0)
plt.xlim(xmin=50)
plt.grid()
plt.show()

plt.figure(figsize=(15,7.5), dpi=600)
for i in range(num_agents): 
    plt.plot(epoch_dur, sat_norm_regret[i], label=label[i])
plt.xlabel("Mean epoch duration")
plt.ylabel("Normalized regret")
#plt.title("Non-Stationary TS")
plt.legend(loc="best")
plt.ylim(ymin=0)
plt.xlim(xmin=50)
plt.grid()
plt.show()