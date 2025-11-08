#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.stats as stats
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from offloading import *
## Plot figures
def plot_figure(test_times, delay, K, M, T, mu, reliability, deadline, sym):
    regret_total = np.zeros((T, test_times))
    real_total = np.zeros((T, M, test_times))
    real_total_theory = np.zeros((T, M, test_times))
    running_time = np.zeros(test_times)
    for m in tqdm (range (test_times), 
               desc="Loading…", 
               ascii=False, ncols=75):
            
        test = offloading(delay, K, M, T, mu, reliability, deadline)
        start_time = time.time() 
        boolean, regret, prob, real, G, count, theory = test.run(T)
        end_time = time.time() 
        regret_total[:,m] = regret
        real_total[:,:,m] = real
        real_total_theory[:,:,m] = theory
        running_time[m] =  end_time - start_time
    
    plot_regret = np.cumsum(regret_total,axis =0)
    x = np.arange(T)
    y = plot_regret.mean(axis=1) 
    confidence_interval = []
    confidence_interval_high=[]
    for i in range(T):
        ci = stats.t.interval(0.95, test_times-1, loc=y[i], scale=stats.sem(plot_regret[i,:]))
        confidence_interval.append(ci[0])
        confidence_interval_high.append(ci[1])
        
    data_soft = {
    'mean_result': plot_regret.mean(axis=1), 
    'confidence_interval': confidence_interval, 
    'confidence_interval_high': confidence_interval_high,
    'mean_running_time': np.mean(running_time),
    'max_running_time': np.max(running_time), 
    'min_running_time': np.min(running_time), 
    }
    df = pd.DataFrame(data_soft)
    df.to_csv(f'Output_T{T}_{test_times}run_{sym}.csv', index=False)    
    print('========================= Plot Regret=========================')
    fig, ax = plt.subplots(1)
    ax.plot(x, y, lw=2, label='O2C algo', color='blue')
    ax.fill_between(x, confidence_interval, confidence_interval_high, facecolor='red', alpha=0.5)
    ax.set_title('Cumulativce regret with 95% confidence interval')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time horizon')
    ax.set_ylabel('Cumulative regret')
    ax.grid()
    plt.savefig(f'Cumulative_regret_{sym}.pdf', format="pdf", bbox_inches="tight")
    
    print('========================= Plot Reliability=========================')
    x1 = np.arange(T)
    plt.figure(figsize=(15,6))
    real_ave = np.mean(real_total, axis=2)
    colors = ['red', 'green', 'blue','black','magenta']
    for user in range(M):
        confidence_interval = []
        confidence_interval_high=[]
        y1 = real_ave[:,user] 
        for i in range(T):
            ci = stats.t.interval(0.95, test_times-1, loc=y1[i], scale=stats.sem(real_total[i,user,:]))
            confidence_interval.append(ci[0])
            confidence_interval_high.append(ci[1])
        plt.plot(x1, y1, color = colors[user], label=f'Plyaer: {user}')
        plt.fill_between(x1,confidence_interval, confidence_interval_high, facecolor= colors[user], alpha=0.5)
        plt.ylim([0.85, 1.01])
        plt.axhline(reliability[user], linestyle='--',color = colors[user], label=f'R^{user} = {reliability[user]}')
        plt.legend(loc='lower right')
        plt.xlabel('Time horizon')
        plt.ylabel('Current Reliability')
        plt.grid()
    plt.savefig(f'reliability_estimation_feasibility_{sym}.pdf', format="pdf", bbox_inches="tight")         
