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
from pymatgen.core.periodic_table import ElementBase, Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from utils_wyckoff import get_tokenized_enc, get_space_group_num_from_letter_enc
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

def from_SLICES(SLICES, strategy=4, fix_duplicate_edge=True):
    """Extract edge_indices, to_jimages and atom_types from decoding a SLICES string.

    Args:
        SLICES (str): SLICES string.
        strategy (int, optional): Strategy number used for encoding SLICES. Defaults to 4.
        fix_duplicate_edge (bool, optional): Flag to indicate whether to fix duplicate edges in 
            SLICES (due to RNN's difficulty in learning long SLICES). Defaults to True.

    Raises:
        Exception: Error: wrong edge indices.
        Exception: Error: wrong edge label.
    """
    atom_types = None
    edge_indices = None
    to_jimages = None
    tokens = SLICES.strip().split(" ")
    if strategy == 4:
        # Find first element symbol to determine where tokenized encoding ends
        first_elem_idx = None
        for i, token in enumerate(tokens):
            try:
                Element(token)
                first_elem_idx = i
                break
            except:
                continue
        #print("first_elem_idx",first_elem_idx)
        if first_elem_idx is None:
            # No element symbols found - invalid SLICES
            raise Exception("Error: no valid element symbols found")
        # Get space group number if tokenized encoding exists
        space_group_num = None
        if first_elem_idx > 0:
            letter_enc = "".join(tokens[:first_elem_idx])
            try:
                space_group_num = get_space_group_num_from_letter_enc(letter_enc)
                #print("space_group_num",space_group_num)
            except:
                space_group_num = None
                raise Exception("Error: space_group_num = None")
        

        # Process atom symbols
        for i in range(first_elem_idx, len(tokens)):
            if tokens[i].isnumeric():
                num_atoms = i - first_elem_idx
                break
        atom_symbols = tokens[first_elem_idx:first_elem_idx+num_atoms]
        
        # Process edges and periodic boundary conditions
        start_idx = first_elem_idx + len(atom_symbols)
        num_edges = int((len(tokens)-start_idx)/3)
        edge_indices = np.zeros([num_edges,2], dtype=int)
        to_jimages = np.zeros([num_edges,3], dtype=int)
        
        for i in range(num_edges):
            edge = tokens[start_idx+i*3:start_idx+(i+1)*3]
            edge_indices[i,0] = int(edge[0])
            edge_indices[i,1] = int(edge[1])
            if edge_indices[i,0] > num_atoms-1 or edge_indices[i,1] > num_atoms-1:
                raise Exception("Error: wrong edge indices")
            for j in range(3):
                if edge[2][j] == '-':
                    to_jimages[i,j] = -1
                elif edge[2][j] == 'o':
                    to_jimages[i,j] = 0
                elif edge[2][j] == '+':
                    to_jimages[i,j] = 1
                else:
                    raise Exception("Error: wrong edge label")

    if fix_duplicate_edge:
        edge_data_ascending = []
        for i in range(len(edge_indices)):
            if edge_indices[i][0] <= edge_indices[i][1]:
                edge_data_ascending.append(list(edge_indices[i])+list(to_jimages[i]))
            else:
                edge_data_ascending.append([edge_indices[i][1],edge_indices[i][0]]+list(np.array(to_jimages[i])*-1))
        edge_data_ascending = np.array(edge_data_ascending, dtype=int)
        edge_data_ascending_unique = np.unique(edge_data_ascending, axis=0)
        edge_indices = edge_data_ascending_unique[:,:2]
        to_jimages = edge_data_ascending_unique[:,2:]

    edge_indices = edge_indices
    to_jimages = to_jimages
    atom_types = np.array([int(PERIODIC_DATA.loc[PERIODIC_DATA["symbol"]==i].values[0][0]) for i in atom_symbols])
    return edge_indices, to_jimages, atom_types, space_group_num

def check_SLICES(SLICES,strategy=4,dupli_check=False,graph_rank_check=True):
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
            edge_indices,to_jimages,atom_types,space_group_num=from_SLICES(SLICES,strategy,fix_duplicate_edge=False)
        else:
            edge_indices,to_jimages,atom_types,space_group_num=from_SLICES(SLICES,strategy,fix_duplicate_edge=True)
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

def get_slices4(atom_symbols, edge_indices, to_jimages, space_group_num=None):
    """Generate SLICES string using strategy 4.
    
    Args:
        atom_symbols (list): List of atomic symbols
        edge_indices (np.array): Edge connectivity indices
        to_jimages (np.array): Periodic boundary conditions
        space_group_num (int, optional): Space group number. If provided, will add tokenized encoding.
            Defaults to None.
        
    Returns:
        str: SLICES string representation
    """
    SLICES = ''
    # Add tokenized encoding if space group number is provided
    if space_group_num is not None:
        tokenized_enc = get_tokenized_enc(space_group_num)
        SLICES = tokenized_enc + ' '
            
    # Add atom symbols
    for i in atom_symbols:
        SLICES += i + ' '
        
    # Add edges and periodic boundary conditions
    for i in range(len(edge_indices)):
        SLICES += str(edge_indices[i][0]) + ' ' + str(edge_indices[i][1]) + ' '
        for j in to_jimages[i]:
            if j <= -1:   # Handle -2, -3, etc (just in case)
                SLICES += '-'
            elif j == 0:
                SLICES += 'o'
            else:   # Handle 2, 3, etc (just in case)
                SLICES += '+'
        SLICES += ' '
    return SLICES

def get_slices_by_strategy(strategy, atom_symbols, edge_indices, to_jimages, space_group_num=None):
    """Convert atomic and edge data to SLICES string using specified strategy.
    
    Args:
        strategy (int): SLICES encoding strategy (1-4)
        atom_symbols (list): List of atomic symbols
        edge_indices (np.array): Edge connectivity indices
        to_jimages (np.array): Periodic boundary conditions
        space_group_num (int, optional): Space group number. If provided, will add tokenized encoding.
            Defaults to None.
        
    Returns:
        str: SLICES string representation
    """
    strategy_method_map = {
        4: get_slices4
    }
    method = strategy_method_map.get(strategy)
    if method:
        return method(atom_symbols, edge_indices, to_jimages, space_group_num)
    else:
        raise ValueError(f"Unknown strategy {strategy}")

def get_canonical_SLICES(SLICES, strategy=4):
    """Convert a SLICES to its canonical form.

    Args:
        SLICES (str): A SLICES string.

    Returns:
        str: The canonical SLICES string.
    """
    def get_index_list_allow_duplicates(ori,mod):
        indexes = defaultdict(deque)
        for i, x in enumerate(mod):
            indexes[x].append(i)
        ids = [indexes[x].popleft() for x in ori]
        return ids

    edge_indices,to_jimages,atom_types,space_group_num=from_SLICES(SLICES,strategy)
    # sort elements
    atom_types_sorted=copy.deepcopy(atom_types)
    atom_types_sorted=np.sort(atom_types_sorted)
    atom_types_sorted=list(atom_types_sorted)
    index_mapping=get_index_list_allow_duplicates(atom_types,atom_types_sorted)
    edge_indices=copy.deepcopy(edge_indices)
    for j in range(len(edge_indices)):
        edge_indices[j][0]=index_mapping[edge_indices[j][0]]
        edge_indices[j][1]=index_mapping[edge_indices[j][1]]
    # sort edges (to facilitate rough edge label sorting)
    edge_indices_asc=copy.deepcopy(edge_indices)
    to_jimages_asc=copy.deepcopy(to_jimages)
    for i in range(len(edge_indices)):
        if edge_indices[i][0]>edge_indices[i][1]:
            edge_indices_asc[i][0]=edge_indices[i][1]
            edge_indices_asc[i][1]=edge_indices[i][0]
            to_jimages_asc[i]=to_jimages[i]*-1
    atom_symbols = [str(ElementBase.from_Z(i)) for i in atom_types_sorted]
    c =np.concatenate((edge_indices_asc, to_jimages_asc), axis=1) 
    sorted_data = np.array(sorted(c, key=lambda x: (x[0], x[1])))
    edge_indices = sorted_data[:, :2]
    to_jimages = sorted_data[:, 2:]
    # sort edge labels
    column_sums = np.sum(to_jimages, axis=0)
    def custom_sort_rule(column): # sum(x*index_x**3)
        weighted_sum=[]
        for i in range(3):
            temp=0
            for j in range(len(column)):
                temp+=(j+1)**3*column[j,i]
            weighted_sum.append(temp)
        return weighted_sum
    sorted_column_indices = np.lexsort((custom_sort_rule(to_jimages), column_sums))
    to_jimages_column_sorted = to_jimages[:, sorted_column_indices]
    # sort edge+ij together
    c =np.concatenate((edge_indices, to_jimages_column_sorted), axis=1) 
    sorted_data = np.array(sorted(c, key=lambda x: (x[0], x[1],x[2],x[3],x[4])))
    edge_indices = sorted_data[:, :2]
    to_jimages = sorted_data[:, 2:]
    return get_slices_by_strategy(strategy, atom_symbols, edge_indices, to_jimages, space_group_num)

def get_coordinates(arc_coord,num_nodes,shortest_path_spanning_graph,spanning):
    """Get fractional coordinates of atoms from fractional coordinates of edge vectors.

    Args:
        arc_coord (np.array): Edge vectors (fractional coords) of a labeled quotient graph.
        num_nodes (int): Number of atoms(nodes) of a labeled quotient graph.
        shortest_path_spanning_graph (list): Shortest path of the spanning graph of a labeled 
            quotient graph.
        spanning (list): Spanning graph of a labeled quotient graph.

    Returns:
        np.array: Fractional coordinates of atoms.
    """
    coordinates=np.zeros((num_nodes,3))  #v0 
    if num_nodes>1:  # !deal with single node case
        for i in range(num_nodes)[1:]:  # get vi
            for h in range(len(shortest_path_spanning_graph[str(i+1)][1:])):    # every edge in shortest path  # str(i) convert '0' to 0
                for j in spanning:            
                    if set(shortest_path_spanning_graph[str(i+1)][h:h+2])==set(j[:2]):  # compare with every edge in spanning tree  
                        if int(shortest_path_spanning_graph[str(i+1)][h])== int(j[0]):                           
                            coordinates[i,:]=coordinates[i,:]+arc_coord[int(j[2][1:])-1]
                        else:
                            coordinates[i,:]=coordinates[i,:]-arc_coord[int(j[2][1:])-1]
    else:
        coordinates[0,:]=[0,0,0]
    return coordinates

def SLICES2space_group_number(SLICES, strategy=4, fix_duplicate_edge=True):
    """
    Convert a SLICES string to its space group number using Eon's graph theory method
    and pymatgen's symmetry analysis. This method reconstructs only the standard 
    placement structure without any optimizations to enhance performance.

    Args:
        SLICES (str): A SLICES string representing the crystal structure.
        strategy (int, optional): Strategy number used for encoding SLICES. Defaults to 4.
        fix_duplicate_edge (bool, optional): Whether to fix duplicate edges in SLICES. Defaults to True.

    Returns:
        int: Space group number of the recreated standard structure.

    Raises:
        Exception: If conversion or space group determination fails.
    """
    try:
        # Step 1: Parse the SLICES string to extract graph information
        edge_indices, to_jimages, atom_types, space_group_num = from_SLICES(SLICES, strategy, fix_duplicate_edge)
        
        # Step 2: Convert graph to network representation
        x_dat, net_voltage = convert_graph(edge_indices, to_jimages)
        net = Net(x_dat, dim=3)
        net.voltage = net_voltage
        
        # Step 3: Analyze graph topology
        net.simple_cycle_basis()
        net.get_lattice_basis()
        net.get_cocycle_basis()
        
        # Step 4: Generate inner product target
        # Use the standard metric tensor without scaling
        net.get_metric_tensor()
        lattice_vectors_std=np.linalg.cholesky(net.metric_tensor)
        #deal with cocycle == none 
        if net.cocycle is not None: 
            net.periodic_rep = np.concatenate((net.cycle_rep, net.cocycle_rep), axis=0)
        else:
            net.periodic_rep=net.cycle_rep
        # Get fractional coordinates of vertices (atoms)
        num_nodes=len(net.graph.nodes)
        G = nx.MultiDiGraph()
        G.add_nodes_from(net.vertices())
        G.add_edges_from(net.all_edges())
        edges=list(G.edges)
        G_nonDi = nx.MultiGraph()
        G_nonDi.add_nodes_from(net.vertices())
        G_nonDi.add_edges_from(net.all_edges())
        mst = tree.minimum_spanning_edges(G_nonDi, algorithm="kruskal", data=False) 
        spanning = list(mst)
            # convert spanning back to MultiDigraph's case
        for i in range(len(spanning)):   
            for j in range(len(edges)):
                if spanning[i][2]==edges[j][2]:
                    spanning[i]=edges[j]
        spanning_graph=nx.MultiGraph()
        spanning_graph.add_nodes_from(G.nodes)
        spanning_graph.add_edges_from(spanning)
        shortest_path = nx.shortest_path(spanning_graph, source='1')
        arc_coord_std=net.lattice_arcs
        inner_p_std = np.dot(np.dot(arc_coord_std, net.metric_tensor), arc_coord_std.T)
        inner_p_std_diag=np.diag(inner_p_std)
        nonzero_edge_index=np.where(inner_p_std_diag>0.0001)[0].tolist()
        if len(nonzero_edge_index) < len(inner_p_std_diag):
            print("unstable graph")
            net.cocycle_rep = np.zeros((net.order-1, net.ndim)) + .5
            net.periodic_rep = np.concatenate((net.cycle_rep, net.cocycle_rep), axis=0)
            arc_coord_std=net.lattice_arcs
        coordinates_std = get_coordinates(
            arc_coord_std, num_nodes, shortest_path, spanning
        )
        
        # Generate atom symbols
        atom_symbols = [str(ElementBase.from_Z(i)) for i in atom_types]
        
        # Create the standard structure using original lattice vectors
        structure_recreated_std = Structure(lattice_vectors_std, atom_symbols, coordinates_std)
        
        # Step 5: Determine the space group number using pymatgen's SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure_recreated_std)
        space_group_num = sga.get_space_group_number()
    except Exception as e:
        print(f"Error in get_space_group_number: {e}")
        return None
    return space_group_num
