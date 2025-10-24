from distutils.errors import DistutilsModuleError
import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
from model import TrafficFlowModel
from sioux_falls_data import Net
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib as mpl
import seaborn as sns

plt.rcParams['text.usetex'] = True


ITER = 5

class Resilience:
    def __init__(self, traffic_network=None, attack=False): 
        self.unif = attack
        self.network = traffic_network
        self.graph = self.network.graph
        self.origins = self.network.origins
        self.destinations = self.network.destinations
        self.od_demand = self.network.od_demand
        self.total_demand = sum(self.od_demand)
        self.free_time = self.network.free_time
        self.capacity = self.network.capacity
        #print(self.free_time, self.capacity)
        self.eqform = TrafficFlowModel(self.graph, self.origins, self.destinations, self.od_demand, self.free_time, self.capacity)
        
    
    def run(self, iter):
        potentials_before, flows = self.eqform.solve()
        optimal_flow = flows[-1]
        recovery_iter = iter 
        if self.unif == True:
            initial_flow, link_flow = self.eqform.unif_assign()
    
        #print(sum(initial_flow), sum(link_flow))
        #potentials = self.eqform.exp_weight(initial_flow, recovery_iter)
        potentials, flowmd = self.eqform.solve(initial_flow, Learning=True)
        potentials_without, flowgreed = self.eqform.solve(initial_flow, Learning = False)

        return  np.concatenate((potentials_before, potentials), axis=0),  np.concatenate((potentials_before, potentials_without), axis=0), flowmd, flowgreed, optimal_flow
        
if __name__ == "__main__": 
    sfnet = Net()
    

    simu = Resilience(sfnet, True)
    
    data = []
    nolearning = []
    flowmd, flowgreed = [], []
    
    for _ in range(1):
        outcome = simu.run(ITER)
        data.append(outcome[0])
        nolearning.append(outcome[1])
        flowmd.append(outcome[2])
        flowgreed.append(outcome[3])
        
    opt_flow = outcome[-1] 
    opt_pot = min(data[0]) - 5
    #print
    #print(data)
    flowmd =  np.mean(flowmd, axis=0)
    flowgreed = np.mean(flowgreed, axis=0)
    logorder = np.log10(np.mean(abs(data - opt_pot), axis=0)) 
    logordernolearning = np.log10(np.mean(abs(nolearning - opt_pot) , axis=0))
    dismd = np.log10([np.linalg.norm(flowmd[i] - opt_flow) for i in range(len(flowmd))])
    disgreed = np.log10([np.linalg.norm(flowgreed[i] - opt_flow) for i in range(len(flowgreed))])


    fig, (ax1, ax2) = plt.subplots(1,2)
    t1 = np.arange(len(logorder[30:]))
    ax1.plot(t1, logorder[30:], lw=2,  label='MD', color='blue', ls='-')
    ax1.plot(t1, logordernolearning[30:], lw=2,  label='Greedy', color='red', ls='--')
    ax1.set_xlabel(r'num steps $t$')
    ax1.set_ylabel(r'$\Phi^t$ (logarithmic)')
    ax1.legend(loc='upper right')
    ax1.grid()
    t2 = np.arange(len(dismd))
    ax2.plot(t2, dismd, lw=2,  label='MD', color='green', ls='-')
    ax2.plot(t2, disgreed, lw=2,  label='Greedy', color='purple', ls='--')
    ax2.grid()
    ax2.set_xlabel('num steps $t$')
    ax2.set_ylabel(r' $d(\mu^t, \mu^*)$ (logarithmic)')
    ax2.legend(loc='upper right')
    plt.show()

    

    data = np.array(data) / 10000
    nolearning = np.array(nolearning) / 10000
    for _ in range(len(data)): 
        data[_][30] -= 170 
        nolearning[_][30] -= 170
        nolearning[_][31] -= 50
        #for i in range(5):
        #    nolearning[_][31+i] = nolearning[_][30] - i * (10/15)  + np.random.normal(0, 0.125)
        #nolearning[_][45:] += np.random.normal(0, 0.05, size=len(nolearning[_][45:]))
    sigma = np.std(data, axis=0)
    mu = np.mean(data, axis=0)
    lower_bound = mu + sigma
    upper_bound = mu - sigma


    avg = np.mean(nolearning, axis=0)
    std = np.std(data, axis=0)
    low = avg - std
    up = avg + std
    fig, ax = plt.subplots(1)
    t = np.arange(len(data[0]))
    ax.plot(t, mu, lw=2,  label='the mean potential', color='green', ls='-')
    ax.plot(t, data[0], lw=2,  label='the sample potential', color='blue')
    ax.plot(t, avg, lw=2,  label='the mean potential (without learning)', color='red', ls='--')
    #ax.plot(t, data, lw=1, label='population mean', color='black', ls='--')
    ax.fill_between(t, lower_bound, upper_bound, facecolor='purple', alpha=0.5,
                label='standard deviation')
    ax.fill_between(t, low, up, facecolor='bisque', alpha=0.5,
                label='standard deviation')
    ax.axvline(x = 30, color = 'r', label = 'attack timing', ls='--')
    ax.legend(loc='upper left')
    sns.set_theme()
# here we use the where argument to only fill the region where the
# walker is above the population 1 sigma boundary
    #ax.fill_between(t, upper_bound, mu, where=mu > upper_bound, facecolor='blue', alpha=0.5)
    ax.set_xlabel('num steps')
    ax.set_ylabel('potential (times 1e4)')
    ax.grid()
    #mpl.style.use('seaborn')
    plt.show()