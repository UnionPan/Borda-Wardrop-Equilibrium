import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
from model import TrafficFlowModel
from sioux_falls_data import Net
import matplotlib.pyplot as plt

ATKITER = 30
M = 5

def proj(a, y):
     
    l = y/a
    idx = np.argsort(l)
    d = len(l)
 
    evalpL = lambda k: np.sum(a[idx[k:]]*(y[idx[k:]] - l[idx[k]]*a[idx[k:]]) ) -1
 
 
    def bisectsearch():
        idxL, idxH = 0, d-1
        L = evalpL(idxL)
        H = evalpL(idxH)
 
        if L<0:
            return idxL
 
        while (idxH-idxL)>1:
            iMid = int((idxL+idxH)/2)
            M = evalpL(iMid)
 
            if M>0:
                idxL, L = iMid, M
            else:
                idxH, H = iMid, M
 
        return idxH
 
    k = bisectsearch()
    lam = (np.sum(a[idx[k:]]*y[idx[k:]])-1)/np.sum(a[idx[k:]])
 
    x = np.maximum(0, y-lam*a)
 
    return x



class Wardrop_atksurf:
    def __init__(self, traffic_network, theta=None, d=None):
        '''A class that defines everything for an attacker to launch an attack to a physical traffic network
           uses zero-th order sampling method to approximate the gradient
        '''
        
        self.network = traffic_network
        if theta == None:
            self.theta = np.eye(self.network.num_of_links)
        else:
            self.theta = theta
        if d == None:
            self.d = np.array(self.network.od_demand/sum(self.network.od_demand))
        else:
            self.d = d
        
        if not self.isstealthy():
            self.stealthy_projection()
        
        #self.false_latency = self.latency_poison(self.theta)
        #self.false_demand = self.demand_poison(self.d)
        self.q_value = None
        self.graph = self.network.graph
        self.origins = self.network.origins
        self.destinations = self.network.destinations
        self.od_demand = self.network.od_demand
        self.total_demand = sum(self.od_demand)
        self.free_time = self.network.free_time
        self.capacity = self.network.capacity
        #print(self.free_time, self.capacity)
        self.eqform = TrafficFlowModel(self.graph, self.origins, self.destinations, self.od_demand, self.free_time, self.capacity)
        self.gamma = 0.9 
        self.lr = 0.01


    def latency_poison(self, theta):
        self.eqform.theta = theta

    def demand_poison(self, d):
        self.eqform.d = d   

    def wardrop_formation(self, theta, d):
        self.latency_poison(theta)
        self.demand_poison(d)
        self.eqform.solve()
        #self.eqform.report()
        return self.eqform._formatted_solution()

    def attk_iter(self):
        
        poa_list= []
        flow_res = []
        path_res = []
        link_vc_res = []
        self.eqform.so_solve()
        for i in range(ATKITER):
            print('Attacking iteration {}'.format(i+1))
            
            theta_grad, d_grad = self.atk_grad()
            self.theta -= self.lr * theta_grad
            self.d -= self.lr * d_grad
            self.stealthy_projection()
            # sample the next iteration Wardrop equilibrium and obtain the value
            poa, link_flow, link_time, path_time, link_vc, so_link_flow, so_link_time, so_path_time, so_link_vc = self.wardrop_formation(self.theta, self.d)

            print(link_flow)
            print('------------')
            print(link_time)
            poa_list.append(poa)
            flow_res.append(link_flow)
            path_res.append(np.mean(path_time))
            link_vc_res.append(link_vc)
            
        return poa_list, flow_res, path_res, link_vc_res 

    def stealthy_projection(self):
        if not self.isstealthy():
            for link in range(self.network.num_of_links):
                self.theta[link] = proj(np.ones(self.network.num_of_links), self.theta[link])
            self.d = proj(np.ones(len(self.network.od_demand)), self.d)

    def isstealthy(self):
        ''' checking if:
            the demand attack vector is valid: sum d = 1 
            theta is column stochastic matrix 
        '''
        sty = False
        if (sum(self.d) - 1 <= 1e-5 and self.d.all() >= 0) and all((sum(self.theta[:,link]) - 1 <= 1e-5 and self.theta[:, link].all() >= 0) for link in range(self.network.num_of_links)):
            sty = True
        return sty

    def atk_grad(self):
        perf_theta = np.zeros(self.theta.shape)
        perf_d = np.zeros(self.d.shape)
        for i in range(M):
            print('sampling for {}th Gaussian smooth'.format(i+1))
            '''we sample both directions for theta and d'''
            U_i_theta = np.random.normal(0, 1, self.theta.shape[0]*self.theta.shape[1]).reshape(*self.theta.shape)
            U_i_d = np.random.normal(0, 1, self.d.shape)

            # perturbing theta using the sampled theta direction
            pert_theta = self.theta + U_i_theta
            # perturbing d using the sampled d direction
            pert_d = self.d + U_i_d

            # projecting the perturbed attacking parameters into the constraint set
            pert_theta_proj = np.zeros(pert_theta.shape)
            for link in range(self.network.num_of_links):
                pert_theta_proj[link] = proj(np.ones(self.network.num_of_links), pert_theta[link])
            pert_d_proj = proj(np.ones(len(self.network.od_demand)), pert_d)

            # get the performance results after perturbing theta
            stackelberg_res_theta = self.wardrop_formation(pert_theta_proj, self.d)
            pert_theta_avgtime = stackelberg_res_theta[1].dot(stackelberg_res_theta[2]) / self.total_demand
            
            # get the performance results after perturbing d
            stackelberg_res_d = self.wardrop_formation(self.theta, pert_d_proj)
            pert_d_avgtime = stackelberg_res_d[1].dot(stackelberg_res_d[2]) / self.total_demand

            # approximate the gradient using the product between system 
            perf_theta += U_i_theta * pert_theta_avgtime
            perf_d += U_i_d * pert_d_avgtime
             
        theta_grad = self.theta - np.eye(self.network.num_of_links) - self.gamma * perf_theta / M
        d_grad = self.d - (self.od_demand / self.total_demand) - self.gamma * perf_d / M
        return theta_grad, d_grad



if __name__ == "__main__":
    sfnet = Net()
    attacker = Wardrop_atksurf(sfnet)
    data = attacker.attk_iter()
    fig, ax = plt.subplots()
    print(data[0])
    ax.plot(np.arange(0,len(data[0]),1), data[0])
    ax.set_title("the PoA evolution")
    ax.set_xlabel("Attack iterations")
    ax.set_ylabel("PoA")
    plt.grid()
    plt.show()
    
    