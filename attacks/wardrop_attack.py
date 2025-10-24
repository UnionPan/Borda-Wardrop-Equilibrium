import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
from model import TrafficFlowModel
from sioux_falls_data import Net
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

ATKITER = 7
M = 3

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
        
        
        #self.false_latency = self.latency_poison(self.theta)
        #self.false_demand = self.demand_poison(self.d)
        self.q_value = None
        self.graph = self.network.graph
        self.num_links = self.network.num_of_links
        self.origins = self.network.origins
        self.destinations = self.network.destinations
        self.od_demand = self.network.od_demand
        self.total_demand = np.sum(self.od_demand)
        self.free_time = self.network.free_time
        self.capacity = self.network.capacity
        #print(self.free_time, self.capacity)
        
        self.gamma = 10 * self.num_links ** 2
        self.lr = 0.1
        
        if theta == None:
            self.theta = np.eye(self.network.num_of_links)
        else:
            self.theta = theta
        if d == None:
            self.d = np.eye(len(self.od_demand))
            #self.d = np.array(self.network.od_demand/sum(self.network.od_demand))
        else:
            self.d = d
        self.eqform = TrafficFlowModel(self.graph, self.origins, self.destinations, self.od_demand, self.free_time, self.capacity, self.theta, self.d)
        
        if not self.isstealthy():
            self.stealthy_projection()

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
        
        
        flow_res = []
        time_res = []
        path_res = []
        link_vc_res = []
        self.eqform.so_solve()
        self.eqform.solve()
        poa = self.eqform.price_of_anarchy()
        poa_list= [poa]
        
        for i in range(ATKITER):
            print('Attacking iteration {}, learning rate {}'.format(i+1, self.lr))

            
            theta_grad, d_grad = self.atk_grad()

            
            if poa >=2 and poa < 3:
                self.lr *= 0.8
            elif poa >=3 and poa < 4:
                self.lr *= 0.7
            elif poa >= 4: #and poa < 5:
                self.lr *= 0.5
            #elif poa >= 5 and poa < 30:
                #self.lr *= 0.1
            #elif poa >=30 and poa < 50:
            #    self.lr *= 0.01
            elif poa > 10:
                break
            #elif (poa >= 10 and poa <=30):
            #    self.lr *= 0.5
            #elif (poa > 30 and poa < 60):
            #    self.lr *= 0.3
            #elif (poa >=60 and poa < 100):
            #    self.lr *= 0.1
            #elif (poa >100):
            #    self.lr *= 0.01
            self.theta -= self.lr  * theta_grad #* (0.95 ** i)
            self.d -= self.lr * d_grad #* (0.95 ** i)
            self.stealthy_projection()

            print('the attack parameter theta is {}'.format(self.theta))
            print('--------------------------------------------------')
            print('the attack parameter d is {}'.format(self.d))
            # sample the next iteration Wardrop equilibrium and obtain the value
            link_flow, link_time, path_time, link_vc = self.wardrop_formation(self.theta, self.d)
            
            poa = self.eqform.price_of_anarchy()
            print('the current link flow is {}'.format(link_flow))
            print('------------')
            print('the current link time is {}'.format(link_time))
            print('----------')
            print('current PoA {}'.format(poa))
            poa_list.append(poa)
            flow_res.append(link_flow)
            time_res.append(link_time)
            path_res.append(np.mean(path_time))
            link_vc_res.append(link_vc)
            

        return poa_list, flow_res, time_res, path_res, link_vc_res 

    def stealthy_projection(self):
        if not self.isstealthy():
            for link in range(self.num_links):
                self.theta[:, link] = proj(np.ones(self.num_links), self.theta[:, link])
            for w in range(len(self.od_demand)):
                self.d[:, w] = proj(np.ones(len(self.od_demand)), self.d[:, w])

    def isstealthy(self):
        ''' checking if:
            the demand attack vector is valid: sum d = 1 
            theta is column stochastic matrix 
        '''
        sty = False
        if all(sum(self.d[:, w]) - 1 <= 1e-5 and self.d[:, w].all() >= 0 for w in range(len(self.od_demand))) and all((sum(self.theta[:,link]) - 1 <= 1e-5 and self.theta[:, link].all() >= 0) for link in range(self.num_links)):
            sty = True
        return sty

    def atk_grad(self):
        perf_theta = np.zeros(self.theta.shape)
        perf_d = np.zeros(self.d.shape)
        for i in range(M):
            print('sampling for {}th Gaussian smooth'.format(i+1))
            '''we sample both directions for theta and d'''
            U_i_theta = np.random.normal(0, 1, self.theta.shape[0]*self.theta.shape[1]).reshape(*self.theta.shape)
            U_i_d = np.random.normal(0, 1, self.d.shape[0]*self.d.shape[1]).reshape(*self.d.shape)

            # perturbing theta using the sampled theta direction
            pert_theta = self.theta + U_i_theta
            # perturbing d using the sampled d direction
            pert_d = self.d + U_i_d

            # projecting the perturbed attacking parameters into the constraint set
            pert_theta_proj = np.zeros(pert_theta.shape)
            pert_d_proj = np.zeros(pert_d.shape)
            
            for link in range(self.num_links):
                pert_theta_proj[:, link] = proj(np.ones(self.num_links), pert_theta[:, link])
            for w in range(len(self.od_demand)):
                pert_d_proj[:, w] = proj(np.ones(len(self.od_demand)), pert_d[:,w])

            # get the performance results after perturbing theta
            stackelberg_res_theta = self.wardrop_formation(pert_theta_proj, self.d)
            pert_theta_avgtime = stackelberg_res_theta[0].dot(stackelberg_res_theta[1]) / self.total_demand
            
            # get the performance results after perturbing d
            stackelberg_res_d = self.wardrop_formation(self.theta, pert_d_proj)
            pert_d_avgtime = stackelberg_res_d[0].dot(stackelberg_res_d[1]) / self.total_demand

            # approximate the gradient using the product between system 
            perf_theta += U_i_theta * pert_theta_avgtime
            perf_d += U_i_d * pert_d_avgtime
        
        sys_perf = self.eqform.so_final_link_flow.dot(self.eqform.so_final_link_time)
        #sys_perf = 1
        #print(sys_perf)
        theta_grad = -  self.gamma * perf_theta / (M * sys_perf)  + 0.05 * (self.theta - np.eye(self.num_links)) 
        d_grad =  - self.gamma * perf_d / (M * sys_perf)  + 0.05  * (self.d - np.eye(len(self.od_demand)))
        
        return theta_grad, d_grad



if __name__ == "__main__":
    sfnet = Net()
    print('link number is {}'.format(sfnet.num_of_links))
    
    attacker = Wardrop_atksurf(sfnet)
    print('the total demand is {}'.format(attacker.total_demand))
    poa, flow, time, path, vc = attacker.attk_iter()
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(0,len(poa),1), poa, marker = "d", markerfacecolor='r')
    ax.set_title("the PPoA evolution")
    ax.set_xlabel("Attack iterations")
    ax.set_ylabel("PPoA")
    plt.grid()
    plt.show()

    so_link_flow, so_link_time, so_path_time, so_link_vc = attacker.eqform.so_formatted_solution()
    #print(time)
    figc, cx = plt.subplots()
    index = np.arange(len(so_link_time))
    bar_width = 0.3
    sorestime = cx.bar(index, so_link_time, bar_width, label='SO')
    werestime = cx.bar(index+bar_width, time[-1], bar_width, label='PWE')
    cx.set_title("Edge Travel Time")
    cx.set_xlabel("Edge Indices")
    cx.set_ylabel("SO v.s. PWE")
    cx.legend()
    #cx.set_xticks(index + bar_width / 2)
    plt.show()

    figd, dx = plt.subplots()

    soresvc = dx.bar(index, so_link_vc, bar_width, label='SO')
    weresvc = dx.bar(index+bar_width, vc[-1], bar_width, label='PWE')
    dx.set_title("Edge Utilization")
    dx.set_xlabel("Edge Indices")
    dx.set_ylabel("SO v.s. PWE")
    dx.legend()
    #dx.set_xticks(index + bar_width / 2)
    plt.show()


'''    
    #link_flow = pd.DataFrame(flow)
    indices = np.arange(0, 5, 4)
    flow = np.array(flow)[indices]
    figb = plt.figure(figsize=(10,10))
    bx = mplot3d.Axes3D(figb)
    dx = 1.2
    dy = .1
    pos = np.arange(flow.shape[1])
    ypos = np.arange(flow.shape[0])
    bx.set_xticks(xpos + dx/2)
    bx.set_yticks(indices)

    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    zpos = np.zeros(flow.shape).flatten()

    dz = flow.ravel()

    bx.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    #bx.w_yaxis.set_ticklabels()
    bx.set_xlabel("edge indices")
    bx.set_ylabel("Attack Iterations")
    bx.set_zlabel("edge flow")
    bx.set_title("The evolution of edge flows")
    plt.show()
'''
    


    
    
    