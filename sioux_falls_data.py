import os
import sys
import pandas as pd
import networkx as nx
import numpy as np

import openmatrix as omx

import matplotlib.pyplot as plt


def import_matrix(matfile):
    f = open(matfile, 'r')
    all_rows = f.read()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        orig=int(orig[0])

        d = [eval('{'+a.replace(';',',').replace(' ','') +'}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations
    zones = max(matrix.keys())
    mat = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            # We map values to a index i-1, as Numpy is base 0
            mat[i, j] = matrix.get(i+1,{}).get(j+1,0)

    index = np.arange(zones) + 1

    myfile = omx.open_file('demand.omx','w')
    myfile['matrix'] = mat
    myfile.create_mapping('taz', index)
    myfile.close()

class Net:
    def __init__(self):
        # Read the file with proper handling of the tab-delimited format
        # The file has leading tabs and trailing semicolons that need to be handled
        with open('sfdata/SiouxFalls_net.tntp', 'r') as f:
            lines = f.readlines()[8:]  # Skip first 8 lines

        # Parse header and data
        header_line = lines[0].strip().strip('~').strip()
        header = [col.strip() for col in header_line.split('\t') if col.strip() and col.strip() != ';']

        # Parse data lines
        data_rows = []
        for line in lines[1:]:
            line = line.strip()
            if line:
                values = [val.strip() for val in line.split('\t') if val.strip() and val.strip() != ';']
                if values:
                    data_rows.append(values)

        self.net = pd.DataFrame(data_rows, columns=header)
        # Convert numeric columns
        for col in ['capacity', 'length', 'free_flow_time', 'b', 'power', 'speed', 'toll', 'link_type']:
            if col in self.net.columns:
                self.net[col] = pd.to_numeric(self.net[col], errors='coerce')
        for col in ['init_node', 'term_node']:
            if col in self.net.columns:
                self.net[col] = pd.to_numeric(self.net[col], errors='coerce').astype(int)

        self.net['edge'] = self.net.index+1


        self.flow = pd.read_csv('sfdata/SiouxFalls_flow.tntp',sep='\t')
        cols_to_drop = [col for col in ['From ', 'To '] if col in self.flow.columns]
        if cols_to_drop:
            self.flow = self.flow.drop(cols_to_drop, axis=1)
        self.flow.rename(columns={"Volume ": "flow", "Cost ": "cost"},inplace=True)

        self.node_coord = pd.read_csv('sfdata/SiouxFalls_node.tntp',sep='\t')
        if ';' in self.node_coord.columns:
            self.node_coord = self.node_coord.drop([';'], axis=1) # Actual Sioux Falls coordinate
        self.node_xy = pd.read_csv('sfdata/SiouxFalls_node_xy.tntp',sep='\t') # X,Y position for good visualization


        self.sioux_falls_df = pd.concat([self.net, self.flow], axis=1)


        self.G = nx.from_pandas_edgelist(self.sioux_falls_df, 'init_node', 'term_node', ['capacity','length','free_flow_time','b','power','speed','toll','link_type','edge', "flow", 'cost'],create_using=nx.MultiDiGraph())
        self.num_of_links = self.G.number_of_edges()


        self.pos_coord = dict([(i,(a,b)) for i, a,b in zip(self.node_coord.Node, self.node_coord.X, self.node_coord.Y)])
        self.pos_xy = dict([(i,(a,b)) for i, a,b in zip(self.node_xy.Node, self.node_xy.X, self.node_xy.Y)])


        self.nodes = pd.read_csv('sfdata/SiouxFalls_node.tntp',sep='\t')
        if ';' in self.nodes.columns:
            self.nodes = self.nodes.drop([';'], axis=1)

        self.graph = [(str(n), [str(nbr) for nbr in nbrdict]) for n, nbrdict in self.G.adjacency()]
        self.free_time = []
        self.capacity = []
        for node, nbrdict in self.G.adjacency():
            for nbr in nbrdict:
                self.free_time.append(nbrdict[nbr][0]['free_flow_time'])
                self.capacity.append(nbrdict[nbr][0]['capacity'])

        self.origins =['14','15','22','23']
        self.destinations = ['4','5','6','8','9','10','11','16','17','18']
        self.od = [(o,d) for o in self.origins for d in self.destinations]
        x = omx.open_file('demand.omx')
        self.od_demand = []
        for o in self.origins:
            for d in self.destinations:
                self.od_demand.append(x['matrix'][int(o),int(d)])
        
    def draw_net(self):
        for n, p in self.pos_coord.items():
            self.G.nodes[n]['pos_coord'] = p
        for n, p in self.pos_xy.items():
            self.G.nodes[n]['pos_xy'] = p

        for node in self.G.nodes:
            if str(node) in self.destinations:
                self.G.nodes[node]['O/D'] = 'destination'
                self.G.nodes[node]['color'] = 'blue'
            elif str(node) in self.origins:
                self.G.nodes[node]['O/D'] = 'origin'
                self.G.nodes[node]['color'] = 'red'
            else:
                self.G.nodes[node]['O/D'] = 'transfer'
                self.G.nodes[node]['color'] = 'green'

        demand=[2000,9000,7000,2000]
        capacity=[5000,4000,6000,5000,4000,4000,4000,4000,1000,5000]
        node_demand=dict([(i,a) for i, a in zip( self.origins,demand)])
        node_capacity=dict([(i,a) for i, a in zip( self.destinations,capacity)])
        for n, p in node_demand.items():
            self.G.nodes[int(n)]['demand'] = p

        for n, p in node_capacity.items():
            self.G.nodes[int(n)]['capacity'] = p
        
        colors=[n[1]['color'] for n in self.G.nodes.data()]      
        ax,fig=plt.subplots(figsize=(12,12))
        nx.draw_networkx(self.G,pos=self.pos_xy,with_labels=True,node_color=colors,arrows=True,arrowsize=20,node_size=800,font_color='white',font_size=14)

        color_node_type= {'red':'Demand Node', 'green': 'Destination Node','blue':'Transfer Node'}
        for c,n in color_node_type.items():
            fig.scatter([],[], c=c, label=n,s=200)
            fig.legend(loc='upper right',fontsize=12)

        fig.set_title('Sioux Falls Network', fontsize=20)
        plt.savefig('siux-falls-network.png')


if __name__ == "__main__":
    sfnet = Net()
    sfnet.draw_net()