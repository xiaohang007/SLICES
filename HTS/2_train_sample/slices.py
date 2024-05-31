import re
import pandas as pd
import re
import networkx as nx
from networkx.algorithms import tree
import numpy as np
import math
import tempfile
import json
from collections import defaultdict, deque
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import logging
import signal,gc
from contextlib import contextmanager
from functools import wraps
import itertools
import copy
from tobascco_net import Net, SystreDB
from config import OFFSET, LJ_PARAMS_LIST, PERIODIC_DATA

def convert_graph(edge_indices,to_jimages):
    """Convert self.edge_indices, self.to_jimages into networkx format.

    Returns:
        list: x_dat.
        list: net_voltage(edge labels).
    """
    edges=list(np.concatenate((edge_indices, to_jimages), axis=1))
    x_dat,net_voltage = [],[]
    for id, (v1, v2, e1, e2, e3) in enumerate(edges):
        ename = "e%i" % (id + 1)
        net_voltage.append((e1, e2, e3))
        x_dat.append((str(v1+1), str(v2+1), dict(label=ename)))  # networkx compliant
    net_voltage=np.array(net_voltage)
    return x_dat, net_voltage

def from_SLICES(SLICES,strategy=3,fix_duplicate_edge=True):
    """Extract edge_indices, to_jimages and atom_types from decoding a SLICES string.

    Args:
        SLICES (str): SLICES string.
        fix_duplicate_edge (bool, optional): Flag to indicate whether to fix duplicate edges in 
        SLICES (due to RNN's difficulty in learning long SLICES). Defaults to False.

    Raises:
        Exception: Error: wrong edge indices.
        Exception: Error: wrong edge label.
    """
    atom_types = None
    edge_indices = None
    to_jimages = None
    tokens=SLICES.split(" ")
    if strategy==3:
        for i in range(len(tokens)):
            if tokens[i].isnumeric():
                num_atoms=i
                break
        atom_symbols=tokens[:num_atoms]
        num_edges=int((len(tokens)-len(atom_symbols))/5)
        edge_indices=np.zeros([num_edges,2],dtype=int)
        to_jimages=np.zeros([num_edges,3],dtype=int)
        for i in range(num_edges):
            edge=tokens[num_atoms+i*5:num_atoms+(i+1)*5]
            edge_indices[i,0]=int(edge[0])
            edge_indices[i,1]=int(edge[1])
            if edge_indices[i,0] > num_atoms-1 or edge_indices[i,1] > num_atoms-1:
                raise Exception("Error: wrong edge indices")
            for j in range(3):
                if edge[j+2]=='-':
                    to_jimages[i,j]=-1
                elif edge[j+2]=='o':
                    to_jimages[i,j]=0
                elif edge[j+2]=='+':
                    to_jimages[i,j]=1
                else:
                    raise Exception("Error: wrong edge label")

    if strategy==1:
        temp_list=[]
        for i in range(len(tokens)):
            if tokens[i].isnumeric():
                temp_list.append(int(tokens[i]))
        num_atoms=max(temp_list)+1
        num_edges=int(len(tokens)/7)
        edge_indices=np.zeros([num_edges,2],dtype=int)
        to_jimages=np.zeros([num_edges,3],dtype=int)
        atom_symbols=['NaN'] * num_atoms
        for i in range(num_edges):
            edge=tokens[i*7:(i+1)*7]
            edge_indices[i,0]=int(edge[2])
            edge_indices[i,1]=int(edge[3])
            atom_symbols[edge_indices[i,0]]=edge[0]
            atom_symbols[edge_indices[i,1]]=edge[1]
            if edge_indices[i,0] > num_atoms-1 or edge_indices[i,1] > num_atoms-1:
                raise Exception("Error: wrong edge indices")
            for j in range(3):
                if edge[j+4]=='-':
                    to_jimages[i,j]=-1
                elif edge[j+4]=='o':
                    to_jimages[i,j]=0
                elif edge[j+4]=='+':
                    to_jimages[i,j]=1
                else:
                    raise Exception("Error: wrong edge label")
        if 'NaN' in atom_symbols:
            raise Exception("Error: wrong atom symbols")

    if strategy==4:
        for i in range(len(tokens)):
            if tokens[i].isnumeric():
                num_atoms=i
                break
        atom_symbols=tokens[:num_atoms]
        num_edges=int((len(tokens)-len(atom_symbols))/3)
        edge_indices=np.zeros([num_edges,2],dtype=int)
        to_jimages=np.zeros([num_edges,3],dtype=int)
        for i in range(num_edges):
            edge=tokens[num_atoms+i*3:num_atoms+(i+1)*3]
            edge_indices[i,0]=int(edge[0])
            edge_indices[i,1]=int(edge[1])
            if edge_indices[i,0] > num_atoms-1 or edge_indices[i,1] > num_atoms-1:
                raise Exception("Error: wrong edge indices")
            for j in range(3):
                if edge[2][j]=='-':
                    to_jimages[i,j]=-1
                elif edge[2][j]=='o':
                    to_jimages[i,j]=0
                elif edge[2][j]=='+':
                    to_jimages[i,j]=1
                else:
                    raise Exception("Error: wrong edge label")            

    if fix_duplicate_edge:
        edge_data_ascending=[]
        for i in range(len(edge_indices)):
            if edge_indices[i][0]<=edge_indices[i][1]:
                edge_data_ascending.append(list(edge_indices[i])+list(to_jimages[i]))
            else:
                edge_data_ascending.append([edge_indices[i][1],edge_indices[i][0]]+list(np.array(to_jimages[i])*-1))
        edge_data_ascending=np.array(edge_data_ascending,dtype=int)
        edge_data_ascending_unique=np.unique(edge_data_ascending,axis=0)
        edge_indices=edge_data_ascending_unique[:,:2]
        to_jimages=edge_data_ascending_unique[:,2:]

    edge_indices=edge_indices
    to_jimages=to_jimages
    atom_types=np.array([int(PERIODIC_DATA.loc[PERIODIC_DATA["symbol"]==i].values[0][0]) for i in atom_symbols])    
    return edge_indices,to_jimages,atom_types

def check_SLICES(SLICES,strategy=3,dupli_check=True,graph_rank_check=True):
    """Check if a slices string conforms to the proper syntax.

    Args:
        SLICES (str): A SLICES string.
        dupli_check (bool, optional): Flag to indicate whether to check if a SLICES has duplicate
            edges. Defaults to True.
        graph_rank_check (bool, optional): A flag that indicates whether to verify if a SLICES corresponds 
        to a crystal graph with a rank H1(X,Z) < 3. The default value is True. It is advisable to set it to
        True for generative AI models and to False for property prediction AI models. In cases where the 
        rank of H1(X,Z) in the graph is less than 3, it may not be possible to reconstruct this SLICES 
        string to the original structure using SLI2Cry. This limitation stems from Eon's method's inability 
        to generate a 3D embedding for a graph with a rank of H1(X,Z) less than 3. For example, if H1(X,Z)=2, 
        then Eon's method can only create a 2D embedding for this graph. However, for property prediction AI 
        models, this limitation is irrelevant since invertibility is not required.

    Returns:
        bool: Return True if a SLICES is syntaxlly valid.
    """
    try:
        if dupli_check:
            edge_indices,to_jimages,atom_types=from_SLICES(SLICES,strategy,fix_duplicate_edge=False)
        else:
            edge_indices,to_jimages,atom_types=from_SLICES(SLICES,strategy,fix_duplicate_edge=True)
    except:
        return False
    #print(edge_indices,to_jimages,atom_types)
    # make sure the rank of first homology group of graph >= 3, in order to get 3D embedding 
    G = nx.MultiGraph()
    G.add_nodes_from([i for i in range(len(atom_types))])
    G.add_edges_from(edge_indices)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
    mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
    b=G.size()-len(list(mst))  # rank of first homology group of graph X(V,E); rank H1(X,Z) = |E| − |E1|
    if b < 3 and graph_rank_check:
        return False
    # check if all nodes has been covered by edges
    nodes_covered=[]
    for i in edge_indices:
        nodes_covered.append(i[0])
        nodes_covered.append(i[1])
    if len(set(nodes_covered))!=len(atom_types):
        return False
    # check if edge labels covers 3 dimension in at least 3 edges, in order to get 3D embedding
    edge_index_covered=[[],[],[]]
    for i in range(len(to_jimages)):
        for j in range(3):
            if to_jimages[i][j]!=0:
                edge_index_covered[j].append(i)
    for i in edge_index_covered:
        if len(i)==0:
            return False
    # check dumplicates(flip)
    if dupli_check:
        edge_data_ascending=[]
        for i in range(len(edge_indices)):
            if edge_indices[i][0]<=edge_indices[i][1]:
                edge_data_ascending.append(list(edge_indices[i])+list(to_jimages[i]))
            else:
                edge_data_ascending.append([edge_indices[i][1],edge_indices[i][0]]+list(np.array(to_jimages[i])*-1))
        def remove_duplicate_arrays(arrays):
            unique_arrays = []
            for array in arrays:
                if array not in unique_arrays:
                    unique_arrays.append(array)
            return unique_arrays
        if len(edge_data_ascending)>len(remove_duplicate_arrays(edge_data_ascending)):
            return False
    # strict case: (still not covering all cases)
    if len(edge_index_covered[1])>=len(edge_index_covered[0]):
        b_sub_a = [i for i in edge_index_covered[1] if i not in edge_index_covered[0]]
    else:
        b_sub_a = [i for i in edge_index_covered[0] if i not in edge_index_covered[1]]
    a_add_b = edge_index_covered[0]+edge_index_covered[1]
    if len(a_add_b)>=len(edge_index_covered[2]):
        c_sub_ab = [i for i in a_add_b if i not in edge_index_covered[2]]
    else:    
        c_sub_ab = [i for i in edge_index_covered[2] if i not in a_add_b]
    if len(b_sub_a)==0 or len(c_sub_ab)==0:
        return False
    try:
        x_dat, net_voltage = convert_graph(edge_indices,to_jimages)
        net = Net(x_dat,dim=3)
        net.voltage = net_voltage
        # check the graph first (super fast)
        net.simple_cycle_basis()
        net.get_lattice_basis()
        net.get_cocycle_basis()
    except Exception as e:
        #print(e)
        return False
    return True
