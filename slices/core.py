# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,subprocess,random,warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XTB_MOD_PATH"] = os.path.abspath(os.path.dirname(__file__))+"/xtb_noring_nooutput_nostdout_noCN"
os.environ["PYTHONWARNINGS"]="ignore" 
from pymatgen.core.structure import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN,BrunnerNN_reciprocal,EconNN,MinimumDistanceNN
from pymatgen.core.periodic_table import ElementBase
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.core.composition import Composition
import re
import networkx as nx
from networkx.algorithms import tree
import numpy as np
from slices.tobascco_net import Net, SystreDB
from slices.config import OFFSET, LJ_PARAMS_LIST, PERIODIC_DATA
import math
import tempfile
import json
from scipy.optimize import fsolve,fmin_l_bfgs_b
from collections import defaultdict, deque
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from m3gnet.models import Relaxer
from chgnet.model import StructOptimizer
from chgnet.model.model import CHGNet
import logging
import tensorflow as tf
import signal,gc
from contextlib import contextmanager
from functools import wraps
import itertools
import copy,sys
import m3gnet.models
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
def function_timeout(seconds: int):
    """Define a decorator that sets a time limit for a function call.

    Args:
        seconds (int): Time limit.

    Raises:
        SystemExit: Timeout exception.

    Returns:
        Decorator: Timeout Decorator.
    """
    def decorator(func):
        @contextmanager
        def time_limit(seconds_):
            def signal_handler(signum, frame):  # noqa
                raise SystemExit("Timed out!")  #TimeoutException
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds_)
            try:
                yield
            finally:
                signal.alarm(0)
        @wraps(func)
        def wrapper(*args, **kwargs):
            with time_limit(seconds):
                return func(*args, **kwargs)
        return wrapper
    return decorator



class SLICES:
    """Invertible Crystal Representation (SLICES and labeled quotient graph)
    """    
    def __init__(self, atom_types=None, edge_indices=None, to_jimages=None, graph_method='crystalnn', check_results=False, optimizer="BFGS",fmax=0.2,steps=100,relax_model="chgnet"):
        """__init__

        Args:
            atom_types (np.array, optional): Atom types in a SLICES string. Defaults to None.
            edge_indices (np.array, optional): Edge indices in a SLICES string. Defaults to None.
            to_jimages (np.array, optional): Edge labels in a SLICES string. Defaults to None.
            graph_method (str, optional): The method used for analyzing the local chemical environments 
                to generate labeled quotient graphs. Defaults to 'econnn'.
            check_results (bool, optional): Flag to indicate whether to output intermediate results for 
                debugging purposes. Defaults to False.
            optimizer (str, optional): Optimizer used in M3GNet_IAP optimization. Defaults to "BFGS".
            fmax (float, optional): Convergence criterion of maximum allowable force on each atom. 
                Defaults to 0.2.
            steps (int, optional): Max steps. Defaults to 100.
        """        
        tf.keras.backend.clear_session()
        gc.collect()
        self.atom_types = atom_types
        self.edge_indices = edge_indices
        self.to_jimages = to_jimages
        self.graph_method = graph_method
        self.check_results = check_results
        self.atom_symbols = None
        self.SLICES = None
        self.unstable_graph = False  # unstable graph flag
        self.fmax=fmax
        self.steps=steps
        self.relax_model=relax_model

        # copy m3gnet model file?
        if self.relax_model=="chgnet":
            with self.suppress_output():
                self.relaxer = StructOptimizer(optimizer_class="BFGS",use_device="cpu")
        if self.relax_model=="m3gnet":
            model_path=m3gnet.models.__path__[0]+'/MP-2021.2.8-EFS/'
            if not os.path.isdir(model_path):
                data_path=os.path.dirname(__file__)+'/MP-2021.2.8-EFS'
                subprocess.call(['mkdir','-p', model_path])
                subprocess.call(['cp',data_path+'/checkpoint',data_path+'/m3gnet.data-00000-of-00001',\
                data_path+'/m3gnet.index',data_path+'/m3gnet.json',model_path])
            self.relaxer = Relaxer(optimizer=optimizer)

    @contextmanager
    def suppress_output(self):
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    def check_element(self):
        """Make sure no atoms with atomic numbers higher than 86 (due to GFN-FF's limitation).

        Returns:
            bool: Return True if all atoms with Z < 87.
        """        
        if self.atom_types.max() < 87:
            return True
        else:
            return False

    def cif2structure_graph(self,string):
        """Convert a cif string to a structure_graph.

        Args:
            string (str): String of a cif file.

        Returns:
            StructureGraph: Pymatgen structure_graph object.
        """
        structure = Structure.from_str(string,'cif')
        if self.graph_method == 'brunnernn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, BrunnerNN_reciprocal())
        elif self.graph_method == 'econnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, EconNN())
        elif self.graph_method == 'mininn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, MinimumDistanceNN())
        elif self.graph_method == 'crystalnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, CrystalNN())
        else:
            print("ERROR - graph_method not implemented") 
        return structure_graph,structure

    def structure2structure_graph(self,structure):
        """Convert a pymatgen structure to a structure_graph.

        Args:
            structure (Structure): A pymatgen Structure.

        Returns:
            StructureGraph: A Pymatgen StructureGraph object.
        """
        if self.graph_method == 'brunnernn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, BrunnerNN_reciprocal())
        elif self.graph_method == 'econnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, EconNN())
        elif self.graph_method == 'mininn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, MinimumDistanceNN())
        elif self.graph_method == 'crystalnn':
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, CrystalNN())
        else:
            print("ERROR - graph_method not implemented") 
        return structure_graph

    def from_SLICES(self,SLICES,strategy=4,fix_duplicate_edge=True):
        """Extract edge_indices, to_jimages and atom_types from decoding a SLICES string.

        Args:
            SLICES (str): SLICES string.
            fix_duplicate_edge (bool, optional): Flag to indicate whether to fix duplicate edges in 
            SLICES (due to RNN's difficulty in learning long SLICES). Defaults to False.

        Raises:
            Exception: Error: wrong edge indices.
            Exception: Error: wrong edge label.
        """
        self.atom_types = None
        self.edge_indices = None
        self.to_jimages = None
        tokens=SLICES.split(" ")
        if strategy==3:
            for i in range(len(tokens)):
                if tokens[i].isnumeric():
                    num_atoms=i
                    break
            self.atom_symbols=tokens[:num_atoms]
            num_edges=int((len(tokens)-len(self.atom_symbols))/5)
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
            self.atom_symbols=['NaN'] * num_atoms
            for i in range(num_edges):
                edge=tokens[i*7:(i+1)*7]
                edge_indices[i,0]=int(edge[2])
                edge_indices[i,1]=int(edge[3])
                self.atom_symbols[edge_indices[i,0]]=edge[0]
                self.atom_symbols[edge_indices[i,1]]=edge[1]
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
            if 'NaN' in self.atom_symbols:
                raise Exception("Error: wrong atom symbols")

        if strategy==4:
            for i in range(len(tokens)):
                if tokens[i].isnumeric():
                    num_atoms=i
                    break
            self.atom_symbols=tokens[:num_atoms]
            num_edges=int((len(tokens)-len(self.atom_symbols))/3)
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

        self.edge_indices=edge_indices
        self.to_jimages=to_jimages
        self.atom_types=np.array([int(PERIODIC_DATA.loc[PERIODIC_DATA["symbol"]==i].values[0][0]) for i in self.atom_symbols])    

    def get_slices_by_strategy(self, strategy, atom_symbols, edge_indices, to_jimages):
        strategy_method_map = {
            1: self.get_slices1,
            2: self.get_slices2,
            3: self.get_slices3,
            4: self.get_slices4
        }
        method = strategy_method_map.get(strategy)
        if method:
            return method(atom_symbols, edge_indices, to_jimages)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    @staticmethod
    def get_slices1(atom_symbols,edge_indices,to_jimages):
        SLICES=""
        for i in range(len(edge_indices)):
            SLICES+=atom_symbols[edge_indices[i][0]]+' '+atom_symbols[edge_indices[i][1]]+' '+str(edge_indices[i][0])+' '+str(edge_indices[i][1])+' '
            for j in to_jimages[i]:
                if j<=-1:
                    SLICES+='- '
                if j==0:
                    SLICES+='o '
                if j>=1:
                    SLICES+='+ '
        return SLICES
    @staticmethod
    def get_slices2(atom_symbols,edge_indices,to_jimages):
        atom_symbols_mod = [ (i+'_')[:2] for i in atom_symbols]
        SLICES=""
        for i in atom_symbols_mod:
            SLICES+=i
        for i in range(len(edge_indices)):
            SLICES+=('0'+str(edge_indices[i][0]))[-2:]+('0'+str(edge_indices[i][1]))[-2:]
            for j in to_jimages[i]:
                if j<=-1:
                    SLICES+='-'
                if j==0:
                    SLICES+='o'
                if j>=1:
                    SLICES+='+'
        return SLICES
    @staticmethod
    def get_slices3(atom_symbols,edge_indices,to_jimages):
        SLICES=''
        for i in atom_symbols:
            SLICES+=i+' '
        for i in range(len(edge_indices)):
            SLICES+=str(edge_indices[i][0])+' '+str(edge_indices[i][1])+' '
            for j in to_jimages[i]:
                if j<=-1:
                    SLICES+='- '
                if j==0:
                    SLICES+='o '
                if j>=1:
                    SLICES+='+ '
        return SLICES
    @staticmethod
    def get_slices4(atom_symbols,edge_indices,to_jimages):
        SLICES=''
        for i in atom_symbols:
            SLICES+=i+' '
        for i in range(len(edge_indices)):
            SLICES+=str(edge_indices[i][0])+' '+str(edge_indices[i][1])+' '
            for j in to_jimages[i]:
                if j<=-1:   # deal with -2, -3, etc (just in case)
                    SLICES+='-'
                if j==0:
                    SLICES+='o'
                if j>=1:    # deal with 2, 3, etc (just in case)
                    SLICES+='+'
            SLICES+=' '
        return SLICES

    def to_SLICES(self,strategy=4):
        """Output a SLICES string based on self.atom_types & self.edge_indices & self.to_jimages.
        Returns:
            str: SLICES string.
        """

        atom_symbols = [str(ElementBase.from_Z(i)) for i in self.atom_types]
        return self.get_slices_by_strategy(strategy,atom_symbols,self.edge_indices,self.to_jimages)

    @staticmethod
    def check_structural_validity(str1):
        """Check the structural validity of a Structure with minimal distance > 0.5 Ang.

        Args:
            str1 (Structure): Input Structure.

        Returns:
            bool: Return True if Structure is structurally valid.
        """
        distance_matrix=str1.lattice.get_all_distances(str1.frac_coords,str1.frac_coords)
        min_value = 10000000 # inf

        rows, cols = len(distance_matrix), len(distance_matrix[0])
        for i in range(rows):
            for j in range(i+1, cols):
                if distance_matrix[i][j] < min_value:
                    min_value = distance_matrix[i][j]
        if min_value <= 0.5:
            return False
        else:
            return True

    def check_SLICES(self,SLICES,strategy=4,dupli_check=False,graph_rank_check=True):
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
                self.from_SLICES(SLICES,strategy,fix_duplicate_edge=False)
            else:
                self.from_SLICES(SLICES,strategy,fix_duplicate_edge=True)
        except:
            return False
        # make sure the rank of first homology group of graph >= 3, in order to get 3D embedding 
        G = nx.MultiGraph()
        G.add_nodes_from([i for i in range(len(self.atom_types))])
        G.add_edges_from(self.edge_indices)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
        b=G.size()-len(list(mst))  # rank of first homology group of graph X(V,E); rank H1(X,Z) = |E| − |E1|
        if b < 3 and graph_rank_check:
            return False
        # check if all nodes has been covered by edges
        nodes_covered=[]
        for i in self.edge_indices:
            nodes_covered.append(i[0])
            nodes_covered.append(i[1])
        if len(set(nodes_covered))!=len(self.atom_types):
            return False
        # check if edge labels covers 3 dimension in at least 3 edges, in order to get 3D embedding
        edge_index_covered=[[],[],[]]
        for i in range(len(self.to_jimages)):
            for j in range(3):
                if self.to_jimages[i][j]!=0:
                    edge_index_covered[j].append(i)
        for i in edge_index_covered:
            if len(i)==0:
                return False
        # check dumplicates(flip)
        if dupli_check:
            edge_data_ascending=[]
            for i in range(len(self.edge_indices)):
                if self.edge_indices[i][0]<=self.edge_indices[i][1]:
                    edge_data_ascending.append(list(self.edge_indices[i])+list(self.to_jimages[i]))
                else:
                    edge_data_ascending.append([self.edge_indices[i][1],self.edge_indices[i][0]]+list(np.array(self.to_jimages[i])*-1))
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
            x_dat, net_voltage = self.convert_graph()
            net = Net(x_dat,dim=3)
            net.voltage = net_voltage
            # check the graph first (super fast)
            net.simple_cycle_basis()
            net.get_lattice_basis()
            net.get_cocycle_basis()
        except:
            return False
        return True

    def check_SLICES_basic(self,SLICES,strategy=4):
        """Check if a slices string conforms to the proper syntax (for encoding only).

        Args:
            SLICES (str): A SLICES string.

        Returns:
            bool: Return True if a SLICES is syntaxlly valid.
        """
        try:
            self.from_SLICES(SLICES,strategy)
        except:
            return False
        # check if all nodes has been covered by edges
        nodes_covered=[]
        for i in self.edge_indices:
            nodes_covered.append(i[0])
            nodes_covered.append(i[1])
        if len(set(nodes_covered))!=len(self.atom_types):
            return False
        return True

    def get_canonical_SLICES(self,SLICES,strategy=4):
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

        self.from_SLICES(SLICES,strategy)
        # sort elements
        atom_types_sorted=copy.deepcopy(self.atom_types)
        atom_types_sorted=np.sort(atom_types_sorted)
        atom_types_sorted=list(atom_types_sorted)
        index_mapping=get_index_list_allow_duplicates(self.atom_types,atom_types_sorted)
        edge_indices=copy.deepcopy(self.edge_indices)
        for j in range(len(self.edge_indices)):
            edge_indices[j][0]=index_mapping[edge_indices[j][0]]
            edge_indices[j][1]=index_mapping[edge_indices[j][1]]
        # sort edges (to facilitate rough edge label sorting)
        edge_indices_asc=copy.deepcopy(edge_indices)
        to_jimages_asc=copy.deepcopy(self.to_jimages)
        for i in range(len(edge_indices)):
            if edge_indices[i][0]>edge_indices[i][1]:
                edge_indices_asc[i][0]=edge_indices[i][1]
                edge_indices_asc[i][1]=edge_indices[i][0]
                to_jimages_asc[i]=self.to_jimages[i]*-1
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
        return self.get_slices_by_strategy(strategy,atom_symbols,edge_indices,to_jimages)

    def SLICES2formula(self,SLICES):
        """Convert a SLICES string to its chemical formula (to facilitate composition screening 
        before SLICES2structure).

        Args:
            SLICES (str): A SLICES string.

        Returns:
            str: Chemical formula.
        """
        match = re.search(r'^(.*?)(\d+)', SLICES)
        extracted_string = match.group(1)
        try:
            composition = Composition(extracted_string)
            formula = composition.formula.replace(' ', '')
            return formula
        except:
            print(SLICES,extracted_string)

    def structure2SLICES(self,structure,strategy=4):
        """Extract edge_indices, to_jimages and atom_types from a pymatgen structure object
         then encode them into a SLICES string.

        Args:
            structure (Structure): A pymatgen Structure.
            strategy (int, optional): Strategy number. Defaults to 3.

        Returns:
            str: A SLICES string.
        """ 
        structure_graph=self.structure2structure_graph(structure)
        atom_types = np.array(structure.atomic_numbers)
        atom_symbols = [str(ElementBase.from_Z(i)) for i in atom_types]
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)
        return self.get_slices_by_strategy(strategy,atom_symbols,edge_indices,to_jimages)

    def structure2SLICESAug(self,structure,strategy=4,num=50):
        """ Convert Structure to SLICES and conduct data augmentation.
        
        (1) extract edge_indices, to_jimages and atom_types from a pymatgen structure object
        (2) encoding edge_indices, to_jimages and atom_types into multiple equalivent SLICES strings 
            with a data augmentation scheme

        Args:
            structure (Structure): A pymatgen Structure.
            strategy (int, optional): Strategy number. Defaults to 3.
            num (int, optional): Increase the dataset size by a magnitude of num. Defaults to 200.

        Returns:
            list: A list of num SLICES strings.
        """
        structure_graph=self.structure2structure_graph(structure)
        atom_types = np.array(structure.atomic_numbers)
        atom_symbols = [str(ElementBase.from_Z(i)) for i in atom_types]
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)
        num_edges=len(edge_indices)
        SLICES_list=[]
        SLICES_list.append(self.get_slices_by_strategy(strategy,atom_symbols,edge_indices,to_jimages))
        #calcualte how many element and edge permuatations needed. round((n/6)**(1/2)) 
        num_permutation=int(math.ceil((num/6)**(1/3)))
        # shuffle to get permu
        permu=[]
        for i in range(num):
            permu.append(tuple(random.sample(atom_symbols, k=len(atom_symbols))))
        permu_unique=list(set(permu))
        # For duplicates, we take the smallest index that has not been taken.
        def get_index_list_allow_duplicates(ori,mod):
            indexes = defaultdict(deque)
            for i, x in enumerate(mod):
                indexes[x].append(i)
            ids = [indexes[x].popleft() for x in ori]
            return ids
        index_mapping=[]
        for i in permu_unique[:num_permutation]:
            index_mapping.append(get_index_list_allow_duplicates(atom_symbols,i))
        def shuffle_dual_list(a,b):
            c = list(zip(a, b))
            random.shuffle(c)
            a2, b2 = zip(*c)
            return a2,b2
        def remove_duplicate_arrays(arrays):
            unique_arrays = []
            for array in arrays:
                if array not in unique_arrays:
                    unique_arrays.append(array)
            return unique_arrays
        # calculate filp list
        flip_list=[]
        flip_list_unique=[]
        for i in range(num):
            flip_list.append(list(np.random.randint(2, size=num_edges)))
        flip_list_unique=remove_duplicate_arrays(flip_list)
        if len(flip_list_unique)>=num_permutation:
            flip_list_unique=flip_list_unique[:num_permutation]
        # shuffle atom list
        for i in range(len(index_mapping)):
            atom_symbols_new=permu_unique[i]
            edge_indices_new=copy.deepcopy(edge_indices)
            for j in range(num_edges):
                edge_indices_new[j][0]=index_mapping[i][edge_indices[j][0]]
                edge_indices_new[j][1]=index_mapping[i][edge_indices[j][1]]
            edge_indices_new_shuffled_list=[]
            to_jimages_shuffled_list=[]
            for j in range(num):
                edge_indices_new_shuffled,to_jimages_shuffled=shuffle_dual_list(edge_indices_new,to_jimages)
                edge_indices_new_shuffled_list.append(edge_indices_new_shuffled)
                to_jimages_shuffled_list.append(to_jimages_shuffled)
            edge_indices_new_shuffled_list_unique = remove_duplicate_arrays(edge_indices_new_shuffled_list)
            to_jimages_shuffled_list_unique = remove_duplicate_arrays(to_jimages_shuffled_list)
            for j in range(min(num_permutation,len(edge_indices_new_shuffled_list_unique))):
                edge_indices_new_final=edge_indices_new_shuffled_list_unique[j]
                to_jimages_shuffled_transposed = [list(x) for x in zip(*to_jimages_shuffled_list_unique[j])]
                to_jimages_shu_trans_per=list(itertools.permutations(to_jimages_shuffled_transposed))
                for k in range(6):
                    to_jimages_shu_trans_per_trans_final=[list(x) for x in zip(*to_jimages_shu_trans_per[k])]
                    # randomly flip edges
                    for l in range(len(flip_list_unique)):
                        edge_indices_new_final_flip=[]
                        to_jimages_shu_trans_per_trans_final_flip=[]
                        for m in range(num_edges):
                            if flip_list_unique[l][m]==1:
                                edge_indices_new_final_flip.append([edge_indices_new_final[m][1],edge_indices_new_final[m][0]])
                                to_jimages_shu_trans_per_trans_final_flip.append(list(np.array(to_jimages_shu_trans_per_trans_final[m])*-1))
                            else:
                                edge_indices_new_final_flip.append(edge_indices_new_final[m])
                                to_jimages_shu_trans_per_trans_final_flip.append(to_jimages_shu_trans_per_trans_final[m])
                        SLICES_list.append(self.get_slices_by_strategy(strategy,atom_symbols_new,edge_indices_new_final_flip,to_jimages_shu_trans_per_trans_final_flip))
        random.shuffle(SLICES_list)
        return SLICES_list[:num]

    def SLICES2SLICESAug_atom_order(self,SLICES,strategy=4,num=10):
        """ Convert Structure to SLICES and conduct data augmentation.
        
        (1) extract edge_indices, to_jimages and atom_types from a pymatgen structure object
        (2) encoding edge_indices, to_jimages and atom_types into multiple equalivent SLICES strings 
            with a data augmentation scheme (only atom_order is randomized)

        Args:
            structure (Structure): A pymatgen Structure.
            strategy (int, optional): Strategy number. Defaults to 3.
            num (int, optional): Increase the dataset size by a magnitude of num. Defaults to 200.

        Returns:
            list: A list of num SLICES strings.
        """

        self.from_SLICES(SLICES,strategy)
        atom_symbols = [str(ElementBase.from_Z(i)) for i in self.atom_types]
        edge_indices=self.edge_indices
        to_jimages=self.to_jimages
        num_edges=len(edge_indices)
        SLICES_list=[]
        SLICES_list.append(self.get_slices_by_strategy(strategy,atom_symbols,edge_indices,to_jimages))
        #calcualte how many element and edge permuatations needed. round((n/6)**(1/2)) 
        num_permutation=num
        # shuffle to get permu
        permu=[]
        for i in range(num*10):
            temp=tuple(random.sample(atom_symbols, k=len(atom_symbols)))
            if temp != tuple(atom_symbols):
                permu.append(temp)
        permu_unique=list(set(permu))


        # For duplicates, we take the smallest index that has not been taken.
        def get_index_list_allow_duplicates(ori,mod):
            indexes = defaultdict(deque)
            for i, x in enumerate(mod):
                indexes[x].append(i)
            ids = [indexes[x].popleft() for x in ori]
            return ids
        index_mapping=[]
        for i in permu_unique[:num_permutation]:
            index_mapping.append(get_index_list_allow_duplicates(atom_symbols,i))
        
        def shuffle_dual_list(a,b):
            c = list(zip(a, b))
            random.shuffle(c)
            a2, b2 = zip(*c)
            return a2,b2
        def remove_duplicate_arrays(arrays):
            unique_arrays = []
            for array in arrays:
                if array not in unique_arrays:
                    unique_arrays.append(array)
            return unique_arrays
        # shuffle atom list
        for i in range(len(index_mapping)):
            atom_symbols_new=permu_unique[i]
            edge_indices_new=copy.deepcopy(edge_indices)
            for j in range(num_edges):
                edge_indices_new[j][0]=index_mapping[i][edge_indices[j][0]]
                edge_indices_new[j][1]=index_mapping[i][edge_indices[j][1]]
            SLICES_list.append(self.get_slices_by_strategy(strategy,atom_symbols_new,edge_indices_new,to_jimages))
        random.shuffle(SLICES_list)
        return SLICES_list[:num]

    def structure2SLICESAug_atom_order(self,structure,strategy=4,num=10):
        """ Convert Structure to SLICES and conduct data augmentation.
        
        (1) extract edge_indices, to_jimages and atom_types from a pymatgen structure object
        (2) encoding edge_indices, to_jimages and atom_types into multiple equalivent SLICES strings 
            with a data augmentation scheme (only atom_order is randomized)

        Args:
            structure (Structure): A pymatgen Structure.
            strategy (int, optional): Strategy number. Defaults to 3.
            num (int, optional): Increase the dataset size by a magnitude of num. Defaults to 200.

        Returns:
            list: A list of num SLICES strings.
        """
        structure_graph=self.structure2structure_graph(structure)
        atom_types = np.array(structure.atomic_numbers)
        atom_symbols = [str(ElementBase.from_Z(i)) for i in atom_types]
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)
        num_edges=len(edge_indices)
        SLICES_list=[]
        SLICES_list.append(self.get_slices_by_strategy(strategy,atom_symbols,edge_indices,to_jimages))
        #calcualte how many element and edge permuatations needed. round((n/6)**(1/2)) 
        num_permutation=num
        # shuffle to get permu
        permu=[]
        for i in range(num*10):
            temp=tuple(random.sample(atom_symbols, k=len(atom_symbols)))
            if temp != tuple(atom_symbols):
                permu.append(temp)
        permu_unique=list(set(permu))


        # For duplicates, we take the smallest index that has not been taken.
        def get_index_list_allow_duplicates(ori,mod):
            indexes = defaultdict(deque)
            for i, x in enumerate(mod):
                indexes[x].append(i)
            ids = [indexes[x].popleft() for x in ori]
            return ids
        index_mapping=[]
        for i in permu_unique[:num_permutation]:
            index_mapping.append(get_index_list_allow_duplicates(atom_symbols,i))
        
        def shuffle_dual_list(a,b):
            c = list(zip(a, b))
            random.shuffle(c)
            a2, b2 = zip(*c)
            return a2,b2
        def remove_duplicate_arrays(arrays):
            unique_arrays = []
            for array in arrays:
                if array not in unique_arrays:
                    unique_arrays.append(array)
            return unique_arrays
        # shuffle atom list
        for i in range(len(index_mapping)):
            atom_symbols_new=permu_unique[i]
            edge_indices_new=copy.deepcopy(edge_indices)
            for j in range(num_edges):
                edge_indices_new[j][0]=index_mapping[i][edge_indices[j][0]]
                edge_indices_new[j][1]=index_mapping[i][edge_indices[j][1]]
            SLICES_list.append(self.get_slices_by_strategy(strategy,atom_symbols_new,edge_indices_new,to_jimages))
        random.shuffle(SLICES_list)
        return SLICES_list[:num]

    def get_dim(self,structure):
        """Get the dimension of a Structure.

        Args:
            structure (Structure): A pymatgen Structure.

        Returns:
            int: The dimension of a Structure.
        """
        if self.graph_method == 'brunnernn':
            bonded_structure = BrunnerNN_reciprocal().get_bonded_structure(structure)
        elif self.graph_method == 'econnn':
            bonded_structure = EconNN().get_bonded_structure(structure)
        elif self.graph_method == 'mininn':
            bonded_structure = MinimumDistanceNN().get_bonded_structure(structure)
        elif self.graph_method == 'crystalnn':
            bonded_structure = CrystalNN().get_bonded_structure(structure)
        else:
            print("ERROR - graph_method not implemented") 
        dim=get_dimensionality_larsen(bonded_structure)
        return dim

    def from_cif(self, string):
        """Extract edge_indices, to_jimages and atom_types from a cif string.

        Args:
            string (str): String of a cif file.
        """
        structure_graph,structure = self.cif2structure_graph(string)
        if self.check_results:
            structure_graph.draw_graph_to_file('sg.png',hide_image_edges = False,node_labels=True)
        self.atom_types = np.array(structure.atomic_numbers)
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)    
        self.edge_indices = np.array(edge_indices)
        self.to_jimages = np.array(to_jimages)

    def structure2crystal_graph_rep(self, structure):
        """convert a pymatgen structure object into the crystal graph representation:
            atom_types, edge_indices, to_jimages.
        Args:
            structure (_type_): _description_

        Returns:
            np.array: Atom types.
            np.array: Edge indices.
            np.array: Edge labels.            
        """
        structure_graph = self.structure2structure_graph(structure)
        if self.check_results:
            structure_graph.draw_graph_to_file('graph.png',hide_image_edges = False)
        atom_types = np.array(structure.atomic_numbers)
        G = nx.MultiGraph()
        G.add_nodes_from(structure_graph.graph.nodes)
        G.add_edges_from(structure_graph.graph.edges)    # convert to MultiGraph (from MultiDiGraph) !MST can only deal with MultiGraph
        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data='to_jimage'):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)    
        return atom_types,np.array(edge_indices),np.array(to_jimages)

    def get_nbf_blist(self):
        """ (1) Get nbf(neighbor list with atom types for xtb_mod).
            (2) Get blist(node indexes of the central unit cell edges in the 3*3*3 supercell). 

        Returns:
            str: nbf.
            np.array: blist.
        """
        if self.atom_types is not None and self.edge_indices is not None and self.to_jimages is not None:
            n_atom=len(self.atom_types)
            voltage=np.concatenate((self.edge_indices, self.to_jimages), axis=1) # voltage is actually [i j volatge]
            voltage_super=[] # [i j volatge] array for the supercell
            for i in range(len(OFFSET)):
                for j in range(len(voltage)):
                    temp=[]
                    temp.append(voltage[j,0]+i*n_atom)   
                    voltage_sum=voltage[j,2:]+OFFSET[i,:]
                    voltage_sum_new=voltage_sum.copy()
                    for k in range(len(voltage_sum)):
                        if voltage_sum[k] == 2:
                            voltage_sum_new[k] = 1
                        elif voltage_sum[k] == -2:
                            voltage_sum_new[k] = -1
                        else:
                            voltage_sum_new[k] = 0
                    target_block = voltage_sum.copy()
                    for k in range(len(voltage_sum)):
                        if voltage_sum[k] == 2:
                            target_block[k] = voltage_sum[k]-3
                        elif voltage_sum[k] == -2:
                            target_block[k] = voltage_sum[k]+3
                    target_block=np.array(target_block)
                    for k in range(len(OFFSET)):
                        if np.array_equal(target_block,OFFSET[k]):
                            target_index=k
                    temp.append(voltage[j,1]+target_index*n_atom)   # calculate the index of the target atom
                    temp.append(voltage_sum_new[0])
                    temp.append(voltage_sum_new[1])
                    temp.append(voltage_sum_new[2])
                    voltage_super.append(temp)
            voltage_super=np.array(voltage_super,dtype=int)
            # get rid of boundary case
            voltage_super_cut=[]
            for i in range(len(voltage_super)):
                if voltage_super[i,2]==0 and voltage_super[i,3]==0 and voltage_super[i,4]==0:
                    temp=[]
                    temp.append(voltage_super[i,0])
                    temp.append(voltage_super[i,1])
                    voltage_super_cut.append(temp)
            voltage_super_cut=np.array(voltage_super_cut,dtype=int)
            # reindex voltage_super_cut due to the removal of atoms with no neighbor
            no_neighbor_index=[]
            for i in range(n_atom*27):
                row,column=np.where(voltage_super_cut==i)
                total=len(row)
                if total==0:   # no neighbor fix (used to lead to Failed to generate charges)
                    no_neighbor_index.append(i)
            atom_symbol_list_super=[]
            atom_symbols=[str(ElementBase.from_Z(i)) for i in self.atom_types]
            for i in range(len(OFFSET)):
                for j in range(n_atom):
                    atom_symbol_list_super.append(atom_symbols[j])
            if 1:  # wheather delete atoms with no neighbor or not
                for i in range(len(voltage_super_cut)):
                    voltage_super_cut[i,0]=voltage_super_cut[i,0]-len(np.where(no_neighbor_index < voltage_super_cut[i,0])[0]) # offset index
                    voltage_super_cut[i,1]=voltage_super_cut[i,1]-len(np.where(no_neighbor_index < voltage_super_cut[i,1])[0])
                for i in range(len(voltage_super)):  # modify voltage_super's index 
                    voltage_super[i,0]=voltage_super[i,0]-len(np.where(no_neighbor_index < voltage_super[i,0])[0]) # offset index
                    voltage_super[i,1]=voltage_super[i,1]-len(np.where(no_neighbor_index < voltage_super[i,1])[0])
                # get atomic symbol list of the supercell
                no_neighbor_index.reverse() # .reverse() only update the original list (no return value)
                for i in no_neighbor_index:
                    del atom_symbol_list_super[i]
            else:
                no_neighbor_index=[]
            # get nbf
            neighbor_list=np.zeros((20,n_atom*27-len(no_neighbor_index)))
            for i in range(n_atom*27-len(no_neighbor_index)):
                row,column=np.where(voltage_super_cut==i)
                total=len(row)
                if total>19:   # deal with cases with more than 19 neighbors 
                    row=row[:19]  
                    column=column[:19]
                for j in range(len(row)):
                    neighbor_list[j,i]=voltage_super_cut[row[j],1-column[j]] + 1   # nbf indexing starts with 1 instead of 0
                neighbor_list[19,i]=total

            nbf=str(n_atom*27-len(no_neighbor_index))+'\n'+' '.join(atom_symbol_list_super)+'\n'
            for i in range(20):
                nbf=nbf+' '.join([str(int(x)) for x in neighbor_list[i,:].tolist()])+'\n'
            # get blist
            blist=voltage_super[len(voltage)*13:len(voltage)*14,:2]+1  # assume that atoms with no neighbors are not in the center cell
        else:
            print("ERROR - crystalgraph is not defined") # cannot generate 3D embedding
        if self.check_results:
            with open('testBonds_cut.top','w') as f:
                f.write(nbf)
            bdict={'blist':blist.tolist()}
            with open('blist.json', 'w') as  f:
                json.dump(bdict,f)
        return nbf, blist

    def get_inner_p_target_debug(self, bond_scaling=1.05):
        """ Get inner product matrix, colattice indices, colattice weights with debug outputs.

        (1) Get inner_p_target(inner_p matrix obtained by gfnff).
        (2) Get colattice_inds(keep track of all the valid colattice dot indices).
        (3) Get colattice_weights(colattice weights for bond or angle).

        Args:
            bond_scaling (float, optional): Bond scaling factor. Defaults to 1.05.

        Returns:
            np.array: Inner product matrix.
            list: Colattice indices.
            list: Colattice weights.
        """
        nbf, blist = self.get_nbf_blist()
        temp_dir = tempfile.TemporaryDirectory(dir="/dev/shm")
        with open(temp_dir.name+'/testBonds_cut.top','w') as f:
            f.write(nbf)
        subprocess.call(os.environ["XTB_MOD_PATH"]+' --gfnff testBonds_cut.top --wrtopo blist,vbond,alist,vangl', \
        cwd=temp_dir.name, shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        if self.check_results:
            os.system("cp "+temp_dir.name+'/testBonds_cut.top '+os.getcwd())
            os.system("cp "+temp_dir.name+'/gfnff_lists.json '+os.getcwd())
        with open(temp_dir.name+'/gfnff_lists.json', 'r') as fgfn:
            txt=fgfn.read()
            txt=txt.replace("********","       0") # deal with the xtb output bug
            data = json.loads(txt)  # read blist,vbond,alist,vangl
            del(txt)
        temp_dir.cleanup()
        blist_original=blist
        blist_flip=np.flip(blist_original,1) # reverse the order
        bond_weight=[]
        blist_super=np.array(data['blist'],dtype=int)
        inner_p_target=np.zeros((len(blist_original),len(blist_original)))
        # bond
        bond_repul_scale=[]  # track repul scale of all bonds
        for i in range(len(blist_original)):
            #matching
            temp1=np.where((blist_super ==blist_original[i] ).all(axis=1))
            temp2=np.where((blist_super ==blist_flip[i] ).all(axis=1))
            if len(temp1[0]):
                index=temp1[0][0]
            elif len(temp2[0]):
                index=temp2[0][0]
            else:
                print('Cannot find bond!!!') 
            if np.isnan(data['vbond'][index][2]):
                bond_weight.append(1)
                data['vbond'][index][2]=1
            else:
                bond_weight.append(abs(data['vbond'][index][2])) # convert back to angstrom
            inner_p_target[i,i]=round((data['vbond'][index][3]*0.529177*1.05*bond_scaling)**2,5)  # *1.04, a empirical parameter
        blist_unique=np.unique(blist_original)
        alist_original=np.array(data['alist'],dtype=int)
        colattice_inds=[[],[]]
        colattice_weights=[]
        for i in range(len(alist_original)):
            ab=alist_original[i][[0,1]] # first bond
            ac=alist_original[i][[0,2]] # second bond
            temp1=np.where((blist_original ==ab ).all(axis=1))
            temp2=np.where((blist_flip ==ab ).all(axis=1))
            temp3=np.where((blist_original ==ac ).all(axis=1))
            temp4=np.where((blist_flip ==ac ).all(axis=1))   
            if (len(temp1[0])+len(temp2[0])) and (len(temp3[0])+len(temp4[0])):
                sign=1
                if len(temp1[0]):
                    index_x=temp1[0][0]
                elif len(temp2[0]):
                    index_x=temp2[0][0]
                    sign=sign*(-1)
                else:
                    print('Cannot find bond1!!!') 
                if len(temp3[0]):
                    index_y=temp3[0][0]
                elif len(temp4[0]):
                    index_y=temp4[0][0]
                    sign=sign*(-1)
                else:
                    print('Cannot find bond2!!!')  
                inner_p_target[index_x,index_y]=sign*math.sqrt(inner_p_target[index_x,index_x])*math.sqrt(inner_p_target[index_y,index_y])*math.cos(data['vangl'][i][0])
                colattice_inds[0].append(int(index_x))
                colattice_inds[1].append(int(index_y))
                colattice_weights.append(abs(data['vangl'][i][1])) # vangl params could be negative numbers
        # colattice_inds weight
        temp_max=0  # angle weight is larger than 0
        for i in range(len(colattice_weights)):
            if colattice_weights[i] > temp_max:
                temp_max=colattice_weights[i]
        for i in range(len(colattice_weights)):
            colattice_weights[i]=round(colattice_weights[i]/temp_max,2)
        temp_max=0 
        for i in range(len(bond_weight)):
            if bond_weight[i] > temp_max:
                temp_max=bond_weight[i]
        for i in range(len(bond_weight)):
            colattice_inds[0].append(i)
            colattice_inds[1].append(i)
            colattice_weights.append(round(bond_weight[i]/temp_max,2))  # angleweight first, bondweight second
        if self.check_results:
            inner_p_target_dict={'inner_p_target':inner_p_target.tolist(),'colattice_inds':colattice_inds,'colattice_weights':colattice_weights}
            with open('inner_p_target.json', 'w') as  f:
                json.dump(inner_p_target_dict,f)
        return inner_p_target, colattice_inds, colattice_weights

    def get_inner_p_target(self, bond_scaling=1.05):
        """ Get inner product matrix, colattice indices, colattice weights.

        (1) Get inner_p_target(inner_p matrix obtained by gfnff).
        (2) Get colattice_inds(keep track of all the valid colattice dot indices).
        (3) Get colattice_weights(colattice weights for bond or angle).

        Args:
            bond_scaling (float, optional): Bond scaling factor. Defaults to 1.05.

        Returns:
            np.array: Inner product matrix.
            list: Colattice indices.
            list: Colattice weights.
        """
        nbf, blist = self.get_nbf_blist()
        temp_dir = tempfile.TemporaryDirectory(dir="/dev/shm")
        try:
            with open(temp_dir.name+'/testBonds_cut.top','w') as f:
                f.write(nbf)
            subprocess.call(os.environ["XTB_MOD_PATH"]+' --gfnff testBonds_cut.top --wrtopo blist,vbond,alist,vangl', \
            cwd=temp_dir.name, shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            if self.check_results:
                os.system("cp "+temp_dir.name+'/testBonds_cut.top '+os.getcwd())
                os.system("cp "+temp_dir.name+'/gfnff_lists.json '+os.getcwd())
            with open(temp_dir.name+'/gfnff_lists.json', 'r') as fgfn:
                txt=fgfn.read()
                txt=txt.replace("********","       0") # deal with the xtb output bug
                data = json.loads(txt)  # read blist,vbond,alist,vangl
                del(txt)
            temp_dir.cleanup()
            blist_original=blist
            blist_flip=np.flip(blist_original,1) # reverse the order
            bond_weight=[]
            blist_super=np.array(data['blist'],dtype=int)
            inner_p_target=np.zeros((len(blist_original),len(blist_original)))
            # bond
            bond_repul_scale=[]  # track repul scale of all bonds
            for i in range(len(blist_original)):
                #matching
                temp1=np.where((blist_super ==blist_original[i] ).all(axis=1))
                temp2=np.where((blist_super ==blist_flip[i] ).all(axis=1))
                if len(temp1[0]):
                    index=temp1[0][0]
                elif len(temp2[0]):
                    index=temp2[0][0]
                else:
                    print('Cannot find bond!!!') 
                if np.isnan(data['vbond'][index][2]):
                    bond_weight.append(1)
                    data['vbond'][index][2]=1
                else:
                    bond_weight.append(abs(data['vbond'][index][2])) # convert back to angstrom
                inner_p_target[i,i]=round((data['vbond'][index][3]*0.529177*1.05*bond_scaling)**2,5)  # *1.04, a empirical parameter
            blist_unique=np.unique(blist_original)
            alist_original=np.array(data['alist'],dtype=int)
            colattice_inds=[[],[]]
            colattice_weights=[]
            for i in range(len(alist_original)):
                ab=alist_original[i][[0,1]] # first bond
                ac=alist_original[i][[0,2]] # second bond
                temp1=np.where((blist_original ==ab ).all(axis=1))
                temp2=np.where((blist_flip ==ab ).all(axis=1))
                temp3=np.where((blist_original ==ac ).all(axis=1))
                temp4=np.where((blist_flip ==ac ).all(axis=1))   
                if (len(temp1[0])+len(temp2[0])) and (len(temp3[0])+len(temp4[0])):
                    sign=1
                    if len(temp1[0]):
                        index_x=temp1[0][0]
                    elif len(temp2[0]):
                        index_x=temp2[0][0]
                        sign=sign*(-1)
                    else:
                        print('Cannot find bond1!!!') 
                    if len(temp3[0]):
                        index_y=temp3[0][0]
                    elif len(temp4[0]):
                        index_y=temp4[0][0]
                        sign=sign*(-1)
                    else:
                        print('Cannot find bond2!!!') 
                    inner_p_target[index_x,index_y]=sign*math.sqrt(inner_p_target[index_x,index_x])*math.sqrt(inner_p_target[index_y,index_y])*math.cos(data['vangl'][i][0])
                    colattice_inds[0].append(int(index_x))
                    colattice_inds[1].append(int(index_y))
                    colattice_weights.append(abs(data['vangl'][i][1])) # vangl params could be negative numbers
            # colattice_inds weight
            temp_max=0  # angle weight is larger than 0
            for i in range(len(colattice_weights)):
                if colattice_weights[i] > temp_max:
                    temp_max=colattice_weights[i]
            for i in range(len(colattice_weights)):
                colattice_weights[i]=round(colattice_weights[i]/temp_max,2)
            temp_max=0 
            for i in range(len(bond_weight)):
                if bond_weight[i] > temp_max:
                    temp_max=bond_weight[i]
            for i in range(len(bond_weight)):
                colattice_inds[0].append(i)
                colattice_inds[1].append(i)
                colattice_weights.append(round(bond_weight[i]/temp_max,2))  # angleweight first, bondweight second
            if self.check_results:
                inner_p_target_dict={'inner_p_target':inner_p_target.tolist(),'colattice_inds':colattice_inds,'colattice_weights':colattice_weights}
                with open('inner_p_target.json', 'w') as  f:
                    json.dump(inner_p_target_dict,f)
            return inner_p_target, colattice_inds, colattice_weights
        except Exception as e:
            print(e)
            temp_dir.cleanup()

    def convert_graph(self):
        """Convert self.edge_indices, self.to_jimages into networkx format.

        Returns:
            list: x_dat.
            list: net_voltage(edge labels).
        """
        edges=list(np.concatenate((self.edge_indices, self.to_jimages), axis=1))
        x_dat,net_voltage = [],[]
        for id, (v1, v2, e1, e2, e3) in enumerate(edges):
            ename = "e%i" % (id + 1)
            net_voltage.append((e1, e2, e3))
            x_dat.append((str(v1+1), str(v2+1), dict(label=ename)))  # networkx compliant
        net_voltage=np.array(net_voltage)
        return x_dat, net_voltage

    @staticmethod 
    def get_uncovered_pair(graph):  # 
        """Get atom pairs not covered by edges of the structure graph. Assuming that all atoms 
            has been included in graph.

        Args:
            graph (Graph): Networkx graph.

        Returns:
            list: Atom pairs not covered by edges of the structure graph.
        """
        num_nodes=len(graph.nodes)
        unique_covered_pair=[]
        covered_pair=[]
        for i in graph.edges:
            covered_pair.append(i[0]+','+i[1])  #int(i[0])-1,int(i[1])-1
        unique_covered_pair=list(dict.fromkeys(covered_pair))
        unique_covered_pair_int=[]
        for i in unique_covered_pair:
            unique_covered_pair_int.append([int(i.split(',')[0])-1,int(i.split(',')[1])-1])
        uncovered_pair=[]
        index_i,index_j=np.triu_indices(num_nodes)
        triu_pair=[]
        for i in range(len(index_i)):
            if index_i[i]!=index_j[i]:
                triu_pair.append([index_i[i],index_j[i]])
        for i in triu_pair:
            if i  not in unique_covered_pair_int:
                uncovered_pair.append(i)
        return uncovered_pair

    def get_uncovered_pair_lj(self,uncovered_pair):
        """Get the lj parameters for atom pairs not covered by edges of the structure graph.

        Args:
            uncovered_pair (list): Atom pairs not covered by edges of the structure graph.

        Returns:
            list: lj parameters for atom pairs not covered by edges of the structure graph.
        """
        uncovered_pair_lj=[]
        # read lj params
        lj_param={}
        for i in LJ_PARAMS_LIST:   
            lj_param[i[0]]=[i[1],i[2]]
        for i in uncovered_pair:
            if i[0]==i[1]:
                uncovered_pair_lj.append(lj_param[self.atom_types[i[0]]])
            else:
                uncovered_pair_lj.append([lj_param[self.atom_types[i[0]]][0]/2+lj_param[self.atom_types[i[0]]][0]/2,\
                lj_param[self.atom_types[i[0]]][1]/2+lj_param[self.atom_types[i[0]]][1]/2])
        return uncovered_pair_lj

    def get_covered_pair_lj(self):
        """Get atom pairs covered by edges of the structure graph stored in self.atom_types.

        Returns:
            list: Atom pairs covered by edges of the structure graph.
        """
        covered_pair_lj=[]
        # read lj params
        lj_param={}
        for i in LJ_PARAMS_LIST:   
            lj_param[i[0]]=[i[1],i[2]]
        for i in self.edge_indices:
            if i[0]==i[1]:
                covered_pair_lj.append(lj_param[self.atom_types[i[0]]])
            else:
                covered_pair_lj.append([lj_param[self.atom_types[i[0]]][0]/2+lj_param[self.atom_types[i[0]]][0]/2,\
                lj_param[self.atom_types[i[0]]][1]/2+lj_param[self.atom_types[i[0]]][1]/2])
        return covered_pair_lj

    def get_rescaled_lattice_vectors(self,inner_p_target,inner_p_std,lattice_vectors_std,arc_coord_std):
        """Get rescaled_lattice_vectors based on the GFNFF bond lengths calculated using the
            topological neighbor list as input.

        Args:
            inner_p_target (np.array): Inner product matrix obtained by GFNFF.
            inner_p_std (np.array): Inner product matrix of the barycentric embedding.
            lattice_vectors_std (np.array): Lattice vectors of the barycentric embedding.
            arc_coord_std (np.array): Edge vectors (fractional coords) of the barycentric embedding.

        Returns:
            np.array: Rescaled lattice vectors.
        """
        scaleList=[]
        inner_p_std_diag=np.diag(inner_p_std)
        nonzero_edge_index=np.where(inner_p_std_diag>0.0001)[0].tolist()
        if len(nonzero_edge_index) < len(inner_p_std_diag):
            self.unstable_graph = True  
        else:
            self.unstable_graph = False  # to refresh this setting in case of Exception
        scale_sum_temp=0
        for i in range(len(self.edge_indices)):
            if i in nonzero_edge_index:
                scaleList.append(inner_p_target[i,i]**(0.5)/inner_p_std[i,i]**(0.5))
                scale_sum_temp+=inner_p_target[i,i]**(0.5)/inner_p_std[i,i]**(0.5)
            else:
                scaleList.append(0) # placeholder for unstable graph
        scale_ave_temp=scale_sum_temp/len(nonzero_edge_index) # assume that at least one ratio is ok
        for i in range(len(self.edge_indices)):
            if i not in nonzero_edge_index:
                scaleList[i]=scale_ave_temp # replace problematic scales with the average of valid values
        weight_sum=[0,0,0]
        scale_sum=[0,0,0]
        for i in range(len(arc_coord_std)):
            weight_sum+=abs(arc_coord_std[i])
            scale_sum+=abs(arc_coord_std[i])*scaleList[i]
        scale_vector=[0,0,0]
        for i in range(len(scale_sum)):
            scale_vector[i]=scale_sum[i]/weight_sum[i]
        lattice_vectors_scaled = np.dot(lattice_vectors_std,np.diag(scale_vector))
        return lattice_vectors_scaled

    @staticmethod 
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

    @staticmethod
    def convert_params(x, ndim, cocycle_size, lattice_type, metric_tensor_std):
        """Extract metric tensor and cocycle rep from x vector.

        Args:
            x (np.array): Ndarray of metric tensor components and colattice vectors.
            ndim (int): Dimensionality of crystal structure corresponding to the labeled quotient graph.
            cocycle_size (int): Size of the cocycle vectors.
            lattice_type (int): Lattice type. 1: a=b=c, 21: a!=b=c, 22: b!=a=c, 23: c!=a=b , 3: a!=b!=c.
            metric_tensor_std (np.array): Metric tensor of the barycentric embedding.

        Returns:
            np.array: Updated metric tensor based on colattice vectors, x.
            np.array: Cocycle rep, the bottom n-1 rows of the alpha matrix.
        """
        if lattice_type==1: 
            cell_lengths = x[:1]
            cocycle = x[1:]
            mt = np.empty((ndim, ndim))
            for i in range(ndim):
                mt[i, i] = x[0]
        elif lattice_type>20:
            cell_lengths = x[:2]
            cocycle = x[2:]
            mt = np.empty((ndim, ndim))
            if lattice_type==21:
                mt[0, 0] = x[0]
                mt[1, 1] = x[1]
                mt[2, 2] = x[1]
            if lattice_type==22:
                mt[0, 0] = x[0]
                mt[1, 1] = x[1]
                mt[2, 2] = x[0]      
            if lattice_type==23:
                mt[0, 0] = x[0]
                mt[1, 1] = x[0]
                mt[2, 2] = x[1]            
        else:
            cell_lengths = x[:3]
            cocycle = x[3:]
            mt = np.empty((ndim, ndim))   
            for i in range(ndim):
                mt[i, i] = x[i]
        g = np.triu_indices(ndim, 1)
        for (i, j) in zip(*g):
            mt[i, j] = metric_tensor_std[i, j]/np.sqrt(metric_tensor_std[i, i])/np.sqrt(metric_tensor_std[j, j])*np.sqrt(mt[i, i])*np.sqrt(mt[j, j])
            mt[j, i] = mt[i, j]                 
        if cocycle_size == 0:
            cocycle_rep = None
        else:
            cocycle_rep = np.reshape(cocycle, (cocycle_size, ndim))
        return mt, cocycle_rep
    
    @staticmethod
    def initialize_x_bounds(ndim,cocycle_rep,metric_tensor_std,lattice_type,delta_theta,delta_x,lattice_expand,lattice_shrink):
        """Initialize x vectors and bounds based on metric_tensor_std, lattice_type and other settings.

        Args:
            ndim (int): Dimensionality of crystal structure corresponding to the labeled quotient graph.
            cocycle_rep (np.array): Cocycle rep, the bottom n-1 rows of the alpha matrix.
            metric_tensor_std (np.array): Metric tensor of the barycentric embedding.
            lattice_type (int): Lattice type. 1: a=b=c, 21: a!=b=c, 22: b!=a=c, 23: c!=a=b , 3: a!=b!=c.
            delta_theta (float): Angle change limit(deprecated! not deleted due to compatibility of HTS 
                scripts, will be deleted in future).
            delta_x (float): Maximum x change allowed.
            lattice_expand (float): Maximum lattice expansion allowed.
            lattice_shrink (float): Maximum lattice shrinkage allowed.

        Returns:
            np.array: Intitial value of x.
            list: Upper and lower bounds of x.
        """
        if lattice_type==1: 
            # equal length 
            mtsize=1
            if cocycle_rep is not None:
                size = int(mtsize + cocycle_rep.shape[0] * ndim)
            else:
                size = int(mtsize)
            x = np.empty(size)
            ub = np.empty(size)
            lb = np.empty(size)
            xinc = 0
            # non triclinic cases, no need to change lattice angles
            x[xinc] = metric_tensor_std[0,0]
            ub[xinc] = metric_tensor_std[0,0]*(1.01*lattice_expand)**2  # max_cell
            lb[xinc] = metric_tensor_std[0,0]*(lattice_shrink)**2  # min_cell
            xinc += 1
        elif lattice_type>20:
            mtsize=2
            if cocycle_rep is not None:
                size = int(mtsize + cocycle_rep.shape[0] * ndim)
            else:
                size = int(mtsize)
            x = np.empty(size)
            ub = np.empty(size)
            lb = np.empty(size)
            if lattice_type==21 or lattice_type==22:
                xinc = 0
                x[xinc] = metric_tensor_std[0,0]
                ub[xinc] = metric_tensor_std[0,0]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[0,0]*(lattice_shrink)**2  # min_cell
                xinc += 1
                x[xinc] = metric_tensor_std[1,1]
                ub[xinc] = metric_tensor_std[1,1]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[1,1]*(lattice_shrink)**2  # min_cell
                xinc += 1
            else:
                xinc = 0
                x[xinc] = metric_tensor_std[0,0]
                ub[xinc] = metric_tensor_std[0,0]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[0,0]*(lattice_shrink)**2  # min_cell
                xinc += 1
                x[xinc] = metric_tensor_std[2,2]
                ub[xinc] = metric_tensor_std[2,2]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[2,2]*(lattice_shrink)**2  # min_cell
                xinc += 1                
        else:
            # triclinic
            mtsize=3
            if cocycle_rep is not None:
                size = int(mtsize + cocycle_rep.shape[0] * ndim)
            else:
                size = int(mtsize)
            x = np.empty(size)
            ub = np.empty(size)
            lb = np.empty(size)
            xinc = 0
            for i in range(3):
                x[xinc] = metric_tensor_std[i,i]
                ub[xinc] = metric_tensor_std[i,i]*(1.01*lattice_expand)**2  # max_cell
                lb[xinc] = metric_tensor_std[i,i]*(lattice_shrink)**2  # min_cell
                xinc += 1
        if cocycle_rep is not None:
            x[xinc:] = cocycle_rep.flatten()
            ub[xinc:] = delta_x
            lb[xinc:] = -1*delta_x
        bounds=[]
        for i in range(len(x)):
            bounds.append((lb[i],ub[i]))        
        return x, bounds

    @staticmethod
    def all_distances(coords1, coords2):
        """Returns the distances between two lists of coordinates
        
        Args:
            coords1: First set of Cartesian coordinates.
            coords2: Second set of Cartesian coordinates.

        Returns:
            np.array: 2d array of Cartesian distances. E.g the distance between
                coords1[i] and coords2[j] is distances[i,j]
        """
        c1 = np.array(coords1)
        c2 = np.array(coords2)
        z = (c1[:, None, :] - c2[None, :, :]) ** 2
        return np.sum(z, axis=-1) ** 0.5

    def func(self,x,ndim,order,mat_target,colattice_inds,colattice_weights, \
        cycle_rep,cycle_cocycle_I,num_nodes,shortest_path,spanning,uncovered_pair, \
        uncovered_pair_lj,covered_pair_lj,vbond_param_ave_covered,vbond_param_ave, \
        lattice_vectors_scaled,structure_species,angle_weight,repul,lattice_type,metric_tensor_std):
        """Objective function: sum squared differences between the inner products of the GFN-FF predicted 
        geometry and the associated inner products (gjk) of the edges in the non-barycentric embedded net.

        Args:
            x (np.array): Ndarray of metric tensor components and colattice vectors.
            ndim (int): Dimensionality of crystal structure corresponding to the labeled quotient graph.
            order (int): Number of nodes of the labeled quotient graph.
            mat_target (np.array): Inner product matrix target calculated with GFNFF predicted geometry.
            colattice_inds (list): keep track of all the valid colattice dot indices.
            colattice_weights (list): Colattice weights for bond or angle.
            cycle_cocycle_I (np.array): The inverse of B matrix.
            num_nodes (int): Number of nodes of the labeled quotient graph(duplicate! not deleted due to 
                compatibility of HTS scripts, will be deleted in future). 
            shortest_path (list): Shortest path of the spanning graph of the labeled quotient graph.
            spanning (list): Spanning graph of the labeled quotient graph.
            uncovered_pair (list): Atom pairs not covered by edges of the structure graph.
            covered_pair_lj (list): lj parameters for atom pairs covered by edges of the structure graph.
            vbond_param_ave_covered (float): Repulsive potential well depth of atom pairs covered by edges 
                of the structure graph. 
            vbond_param_ave (float): Repulsive potential well depth of atom pairs not covered by edges of
                the structure graph.
            structure_species (list): Atom symbols of the labeled quotient graph.
            angle_weight (float): Weight of angular terms in the object function.
            repul (bool): Flag to indicate whether repulsive potential is considered in the object function.
            lattice_type (int): Lattice type. 1: a=b=c, 21: a!=b=c, 22: b!=a=c, 23: c!=a=b , 3: a!=b!=c.
            metric_tensor_std (np.array): Metric tensor of the barycentric embedding.

        Returns:
            float: Value of the object function.
        """
        square_diff=0
        # convert x to inner_p
        metric_tensor, cocycle_rep = self.convert_params(x, ndim, int(order - 1),lattice_type,metric_tensor_std)
        if cocycle_rep is not None: 
            periodic_rep = np.concatenate((cycle_rep, cocycle_rep))
        else:
            periodic_rep=cycle_rep
        lattice_arcs = np.dot(cycle_cocycle_I, periodic_rep)
        inner_p = np.dot(np.dot(lattice_arcs, metric_tensor), lattice_arcs.T)

        for k in range(len(colattice_inds[0])):
            i=colattice_inds[0][k]
            j=colattice_inds[1][k]
            if i==j:
                square_diff += (inner_p[i][j]-mat_target[i][j])**2 + 4*vbond_param_ave_covered*(covered_pair_lj[i][1]/np.sqrt(inner_p[i][j]))**12  # inner_p单位是 Angstrom的平方
            else:
                square_diff += angle_weight*colattice_weights[k]*(inner_p[i][j]-mat_target[i][j])**2  # divide angle weight by 3
        # repulsive part of LJ potential to prevent atom collision
        if repul:
            # calculate distance matrix
            coordinates_temp=self.get_coordinates(lattice_arcs,num_nodes,shortest_path,spanning)
            coordinates_temp_cart=np.dot(coordinates_temp,lattice_vectors_scaled)  # using the orginal scaled lattice vectors to simplify calculation
            distance_matrix=self.all_distances(coordinates_temp_cart,coordinates_temp_cart)
            for i in range(len(uncovered_pair)):  #  cutoff        # epsilon       # sigma\
                r_temp=distance_matrix[uncovered_pair[i][0],uncovered_pair[i][1]]
                if r_temp<uncovered_pair_lj[i][0]:
                    square_diff+=4*vbond_param_ave*(uncovered_pair_lj[i][1]/r_temp)**12 
        return square_diff

    def func_check(self,x,ndim,order,mat_target,colattice_inds,colattice_weights,cycle_rep,cycle_cocycle_I,num_nodes,shortest_path,spanning,uncovered_pair,uncovered_pair_lj,covered_pair_lj,vbond_param_ave_covered,vbond_param_ave,lattice_vectors_scaled,structure_species,angle_weight,repul,lattice_type,metric_tensor_std):
        """Objective function: sum squared differences between the inner products of the GFN-FF predicted 
        geometry and the associated inner products (gjk) of the edges in the non-barycentric embedded net.
        This version of func() will output debug info.

        Args:
            x (np.array): Ndarray of metric tensor components and colattice vectors.
            ndim (int): Dimensionality of crystal structure corresponding to the labeled quotient graph.
            order (int): Number of nodes of the labeled quotient graph.
            mat_target (np.array): Inner product matrix target calculated with GFNFF predicted geometry.
            colattice_inds (list): keep track of all the valid colattice dot indices.
            colattice_weights (list): Colattice weights for bond or angle.
            cycle_cocycle_I (np.array): The inverse of B matrix.
            num_nodes (int): Number of nodes of the labeled quotient graph(duplicate! not deleted due to 
                compatibility of HTS scripts, will be deleted in future). 
            shortest_path (list): Shortest path of the spanning graph of the labeled quotient graph.
            spanning (list): Spanning graph of the labeled quotient graph.
            uncovered_pair (list): Atom pairs not covered by edges of the structure graph.
            covered_pair_lj (list): lj parameters for atom pairs covered by edges of the structure graph.
            vbond_param_ave_covered (float): Repulsive potential well depth of atom pairs covered by edges 
                of the structure graph. 
            vbond_param_ave (float): Repulsive potential well depth of atom pairs not covered by edges of
                the structure graph.
            structure_species (list): Atom symbols of the labeled quotient graph.
            angle_weight (float): Weight of angular terms in the object function.
            repul (bool): Flag to indicate whether repulsive potential is considered in the object function.
            lattice_type (int): Lattice type. 1: a=b=c, 21: a!=b=c, 22: b!=a=c, 23: c!=a=b , 3: a!=b!=c.
            metric_tensor_std (np.array): Metric tensor of the barycentric embedding.

        Returns:
            float: Value of the object function.
        """
        square_diff=0
        # convert x to inner_p
        metric_tensor, cocycle_rep = self.convert_params(x, ndim, int(order - 1),lattice_type,metric_tensor_std)
        if cocycle_rep is not None: 
            periodic_rep = np.concatenate((cycle_rep, cocycle_rep))
        else:
            periodic_rep=cycle_rep
        lattice_arcs = np.dot(cycle_cocycle_I, periodic_rep)
        inner_p = np.dot(np.dot(lattice_arcs, metric_tensor), lattice_arcs.T)
        # calculate distance matrix
        coordinates_temp=self.get_coordinates(lattice_arcs,num_nodes,shortest_path,spanning)
        coordinates_temp_cart=np.dot(coordinates_temp,lattice_vectors_scaled)  # using the orginal scaled lattice vectors to simplify calculation
        distance_matrix=self.all_distances(coordinates_temp_cart,coordinates_temp_cart)
        for k in range(len(colattice_inds[0])):
            i=colattice_inds[0][k]
            j=colattice_inds[1][k]
            if i==j:
                square_diff += (inner_p[i][j]-mat_target[i][j])**2 + 4*vbond_param_ave_covered*(covered_pair_lj[i][1]/np.sqrt(inner_p[i][j]))**12  # inner_p单位是 Angstrom的平方
                print("bond:"+str(i)+','+str(j)+','+str(round(np.sqrt(inner_p[i][j]),2))+','+str(round(np.sqrt(mat_target[i][j]),2))+','+str(round((inner_p[i][j]-mat_target[i][j])**2,5))+','+str(round(4*vbond_param_ave_covered*(covered_pair_lj[i][1]/np.sqrt(inner_p[i][j]))**12,5)))
            else:
                square_diff += angle_weight*colattice_weights[k]*(inner_p[i][j]-mat_target[i][j])**2  # divide angle weight by 3
                print("angle:"+str(i)+','+str(j)+','+str(round(inner_p[i][j]/np.sqrt(inner_p[i][i])/np.sqrt(inner_p[j][j]),5))+','+str(round(mat_target[i][j]/np.sqrt(mat_target[i][i])/np.sqrt(mat_target[j][j]),5))+','+str(round(angle_weight*colattice_weights[k]*(inner_p[i][j]-mat_target[i][j])**2,5)))
        # repulsive part of LJ potential to prevent atom collision
        if repul:
            for i in range(len(uncovered_pair)):  #  cutoff        # epsilon       # sigma\
                r_temp=distance_matrix[uncovered_pair[i][0],uncovered_pair[i][1]]
                if r_temp<uncovered_pair_lj[i][0]:
                    square_diff+=4*vbond_param_ave*(uncovered_pair_lj[i][1]/r_temp)**12 
                    print("repul:"+str(uncovered_pair[i][0])+','+str(uncovered_pair[i][1])+','+str(round(r_temp,4))+','+str(round(4*vbond_param_ave*(uncovered_pair_lj[i][1]/r_temp)**12,4)))
        return square_diff

    def to_structures(self, bond_scaling=1.05, delta_theta=0.005, delta_x=0.45,lattice_shrink=1,lattice_expand=1.25,angle_weight=0.5,vbond_param_ave_covered=0.00,vbond_param_ave=0.01,repul=True):        
        """The inverse transform of the crystal graph of a SLICES string to its crystal structure.
        Convert edge_indices, to_jimages and atom_types stored in the InvCryRep instance back to 
        3 pymatgen structure objects and the energy per atom predicted with M3GNet.

        (1) barycentric embedding net with rescaled lattice based on the average bond scaling factors
         calculated with modified GFN-FF predicted geometry 
        (2) non-barycentric net embedding that matches bond lengths and bond angles predicted with
         modified GFN-FF
        (3) non-barycentric net embedding undergone cell optimization using M3GNet IAPs
        if cell optimization failed, then output (1) and (2)

        Args:
            bond_scaling (float, optional): Bond scaling factor. Defaults to 1.05.
            delta_theta (float): Angle change limit(deprecated! not deleted due to compatibility of HTS 
                scripts, will be deleted in future).
            delta_x (float, optional): Maximum x change allowed. Defaults to 0.45.
            lattice_shrink (int, optional): Maximum lattice shrinkage allowed. Defaults to 1.
            lattice_expand (float, optional): Maximum lattice expansion allowed. Defaults to 1.25.
            angle_weight (float, optional): Weight of angular terms in the object function. Defaults to 0.5.
            vbond_param_ave_covered (float, optional): Repulsive potential well depth of atom pairs covered 
                by edges of the structure graph. Defaults to 0.00.
            vbond_param_ave (float, optional): Repulsive potential well depth of atom pairs not covered by 
                edges of the structure graph. Defaults to 0.01.
            repul (bool, optional): Flag to indicate whether repulsive potential is considered in the object 
                function. Defaults to True.

        Returns:
            list: [Rescaled Structure, ZL*-optimized Structure,  IAP-optimized Structure]
            float: Energy per atom predicted with M3GNet.
        """
        x_dat, net_voltage = self.convert_graph()
        net = Net(x_dat,dim=3)
        net.voltage = net_voltage
        if self.check_results:
            fig = plt.figure()
            nx.draw(net.graph, ax=fig.add_subplot(111))
            fig.savefig("graph.png")
        # check the graph first (super fast)
        net.simple_cycle_basis()
        net.get_lattice_basis()
        net.get_cocycle_basis()
        # then calculate inner_p_target (slower)
        if self.check_results:
            inner_p_target, colattice_inds, colattice_weights = self.get_inner_p_target_debug(bond_scaling)
        else:
            inner_p_target, colattice_inds, colattice_weights = self.get_inner_p_target(bond_scaling)
        uncovered_pair = self.get_uncovered_pair(net.graph)
        uncovered_pair_lj = self.get_uncovered_pair_lj(uncovered_pair)
        covered_pair_lj = self.get_covered_pair_lj()
        #deal with cocycle == none 
        if net.cocycle is not None: 
            net.periodic_rep = np.concatenate((net.cycle_rep, net.cocycle_rep), axis=0)
        else:
            net.periodic_rep=net.cycle_rep
        net.get_metric_tensor()
        lattice_vectors_std=np.linalg.cholesky(net.metric_tensor)
        arc_coord_std=net.lattice_arcs
        # get spanning and shortest_path_spanning_graph
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
        # calculate anisotropic scaled lattice vectors of standard placement
        inner_p_std = np.dot(np.dot(net.lattice_arcs, net.metric_tensor), net.lattice_arcs.T)
        lattice_vectors_scaled = self.get_rescaled_lattice_vectors(inner_p_target,inner_p_std,lattice_vectors_std,arc_coord_std)
        metric_tensor_std=np.dot(lattice_vectors_scaled,lattice_vectors_scaled.T)
        # add random pertubation to cocycle_rep to deal with unstable graph
        if self.unstable_graph: 
            net.cocycle_rep = np.zeros((net.order-1, net.ndim)) + .5
            #net.cocycle_rep = np.random.random((net.order-1, net.ndim)) - .5
            net.periodic_rep = np.concatenate((net.cycle_rep, net.cocycle_rep), axis=0)
            arc_coord_std=net.lattice_arcs
        # get the fractional coordinates of vertices of standard placement
        coordinates_std=self.get_coordinates(arc_coord_std,num_nodes,shortest_path,spanning) 
        # get the gfnff-scaled standard placement
        atom_symbols=[str(ElementBase.from_Z(i)) for i in self.atom_types]
        structure_recreated_std = Structure(lattice_vectors_scaled, atom_symbols,coordinates_std)
        # optimize X (lattice vectors and cocycle_rep)
        # get lattice type
        lattice_length_list=[metric_tensor_std[0,0],metric_tensor_std[1,1],metric_tensor_std[2,2]]
        lattice_length_list_unique=list(set(lattice_length_list))
        lattice_type=len(lattice_length_list_unique)
        if lattice_type==2:
            if metric_tensor_std[0,0]==metric_tensor_std[1,1]:
                lattice_type=23
            if metric_tensor_std[0,0]==metric_tensor_std[2,2]:
                lattice_type=22
            if metric_tensor_std[2,2]==metric_tensor_std[1,1]:
                lattice_type=21
        x,bounds = self.initialize_x_bounds(net.ndim,net.cocycle_rep,metric_tensor_std,lattice_type,delta_theta,delta_x,lattice_expand,lattice_shrink)
        x=fmin_l_bfgs_b(self.func, x, fprime=None, args= \
        (net.ndim,net.order,inner_p_target,colattice_inds,colattice_weights,net.cycle_rep,net.cycle_cocycle_I, \
        num_nodes,shortest_path,spanning,uncovered_pair,uncovered_pair_lj,covered_pair_lj,vbond_param_ave_covered,vbond_param_ave, \
        lattice_vectors_scaled,atom_symbols,angle_weight,repul,lattice_type,metric_tensor_std), \
        approx_grad=True, bounds=bounds, m=10, factr=10000000.0, pgtol=1e-05, \
        epsilon=1e-08, iprint=- 1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
        #get optimized structure
        net.metric_tensor, net.cocycle_rep = self.convert_params(x[0], net.ndim, int(net.order - 1),lattice_type,metric_tensor_std)
        lattice_vectors_new=np.linalg.cholesky(net.metric_tensor)
        if net.cocycle is not None: 
            net.periodic_rep = np.concatenate((net.cycle_rep, net.cocycle_rep), axis=0)
        else:
            net.periodic_rep=net.cycle_rep
        arc_coord_new=net.lattice_arcs
        coordinates_new=self.get_coordinates(arc_coord_new,num_nodes,shortest_path,spanning) 
        structure_recreated_opt = Structure(lattice_vectors_new, atom_symbols,coordinates_new)
        if self.check_results:
            print(x[0])
            print(self.func_check(x[0],net.ndim,net.order,inner_p_target,colattice_inds,colattice_weights,net.cycle_rep,net.cycle_cocycle_I, \
            num_nodes,shortest_path,spanning,uncovered_pair,uncovered_pair_lj,covered_pair_lj,vbond_param_ave_covered,vbond_param_ave, \
            lattice_vectors_scaled,atom_symbols,angle_weight,repul,lattice_type,metric_tensor_std))
        try:
            if num_nodes <= 20:
                structure_recreated_opt2, final_energy_per_atom=self.relax(structure_recreated_opt)
            elif 20 < num_nodes <= 40:
                structure_recreated_opt2, final_energy_per_atom=self.relax_large_cell1(structure_recreated_opt)
            else:
                structure_recreated_opt2, final_energy_per_atom=self.relax_large_cell2(structure_recreated_opt)                                
            return [structure_recreated_std, structure_recreated_opt,  structure_recreated_opt2 ],final_energy_per_atom
        except Exception as e:
            print(e)
            return [structure_recreated_std, structure_recreated_opt],0

    def SLICES2structure(self,SLICES,strategy=4,fix_duplicate_edge=True):
        """Convert a SLICES string back to its original crystal structure.

        Args:
            SLICES (str): A SLICES string.

        Returns:
            Structure: A pymatgen Structure object.
            float: Energy per atom predicted with M3GNet.
        """
        self.from_SLICES(SLICES,strategy,fix_duplicate_edge)
        structures,final_energy_per_atom = self.to_structures()
        return structures[-1],final_energy_per_atom

    def to_relaxed_structure(self, bond_scaling=1.05, delta_theta=0.005, delta_x=0.45,lattice_shrink=1,lattice_expand=1.25,angle_weight=0.5,vbond_param_ave_covered=0.000,vbond_param_ave=0.01,repul=True):
        """
        Convert edge_indices, to_jimages and atom_types stored in the InvCryRep instance back to 
        a pymatgen structure object: non-barycentric net embedding undergone cell optimization 
        using M3GNet IAPs.
        If cell optimization failed, then raise error.
        """
        structures,final_energy_per_atom=self.to_structures(bond_scaling,delta_theta,delta_x,lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul)
        if len(structures)==3:        
            return structures[-1],final_energy_per_atom
        else:
            raise Exception("relax failed")

    def to_4structures(self, bond_scaling=1.05, delta_theta=0.005, delta_x=0.45,lattice_shrink=1,lattice_expand=1.25,angle_weight=0.5,vbond_param_ave_covered=0.000,vbond_param_ave=0.01,repul=True):
        """
        Designed for benchmark.
        """
        structures,final_energy_per_atom=self.to_structures(bond_scaling,delta_theta,delta_x,lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul)
        try:
            std2, final_energy_per_atom=self.relax(structures[0])
            return [structures[0], std2,  structures[1],structures[2] ],final_energy_per_atom
        except Exception as e:
            print(e)
            return structures,final_energy_per_atom

    @function_timeout(seconds=180)
    def relax(self,struc):
        """Cell optimization using CHGNET/M3GNET IAPs (time limit is set to 60 seconds 
        to prevent buggy cell optimization that takes fovever to finish).

        Args:
            struc (Structure): A pymatgen Structure object.

        Returns:
            Structure: Optimized pymatgen Structure object with CHGNET/M3GNET IAP.
            float: Energy per atom predicted with CHGNET/M3GNET.
        """
        if self.check_results:
            relax_results = self.relaxer.relax(struc,fmax=self.fmax,steps=self.steps)
        else:
            with self.suppress_output():
                relax_results = self.relaxer.relax(struc,fmax=self.fmax,steps=self.steps)
        final_structure = relax_results['final_structure']
        final_energy_per_atom = float(relax_results['trajectory'].energies[-1] / len(struc))
        return final_structure,final_energy_per_atom

    @function_timeout(seconds=360)
    def relax_large_cell1(self,struc):
        """Cell optimization using CHGNET/M3GNET IAPs (time limit is set to 360 seconds 
        to prevent buggy cell optimization that takes fovever to finish).

        Args:
            struc (Structure): A pymatgen Structure object.

        Returns:
            Structure: Optimized pymatgen Structure object with CHGNET/M3GNET IAP.
            float: Energy per atom predicted with CHGNET/M3GNET.
        """
        if self.check_results:
            relax_results = self.relaxer.relax(struc,fmax=self.fmax,steps=self.steps)
        else:
            with self.suppress_output():
                relax_results = self.relaxer.relax(struc,fmax=self.fmax,steps=self.steps)
        final_structure = relax_results['final_structure']
        final_energy_per_atom = float(relax_results['trajectory'].energies[-1] / len(struc))
        return final_structure,final_energy_per_atom

    @function_timeout(seconds=1000)
    def relax_large_cell2(self,struc):
        """Cell optimization using CHGNET/M3GNET IAPs (time limit is set to 1000 seconds 
        to prevent buggy cell optimization that takes fovever to finish).

        Args:
            struc (Structure): A pymatgen Structure object.

        Returns:
            Structure: Optimized pymatgen Structure object with CHGNET/M3GNET IAP.

            float: Energy per atom predicted with CHGNET/M3GNET.
        """
        if self.check_results:
            relax_results = self.relaxer.relax(struc)
        else:
            with self.suppress_output():
                relax_results = self.relaxer.relax(struc)
        final_structure = relax_results['final_structure']
        final_energy_per_atom = float(relax_results['trajectory'].energies[-1] / len(struc))
        return final_structure,final_energy_per_atom

    def match_check(self,ori,opt,std,ltol=0.2, stol=0.3, angle_tol=5):
        """ (1) Calculate match rates of structure (2) and (1) with respect to the 
                original structure.
            (2) Calculate topological(Jaccard) distances of structuregraph of structure
                (2) and (1) with respect to structuregraph of the original structure.
        """
        ori_checked=Structure.from_str(ori.to(fmt="poscar"),"poscar") 
        opt_checked=Structure.from_str(opt.to(fmt="poscar"),"poscar")
        std_checked=Structure.from_str(std.to(fmt="poscar"),"poscar")
        sg_ori=self.structure2structure_graph(ori_checked)
        sg_opt=self.structure2structure_graph(opt_checked)
        sg_std=self.structure2structure_graph(std_checked)
        sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, scale=True, attempt_supercell=False, comparator=ElementComparator())
        return str(int(sm.fit(ori_checked, opt_checked))),str(int(sm.fit(ori_checked, std_checked))),str(sg_ori.diff(sg_opt,strict=False)["dist"]),str(sg_ori.diff(sg_std,strict=False)["dist"])

    def match_check3(self,ori,opt2,opt,std,ltol=0.2, stol=0.3, angle_tol=5):
        """ (1) Calculate match rates of structure (3), (2) and (1) with 
                respect to the original structure.
            (2) Calculate topological distances of structuregraph of structure (3), (2)
                and (1) with respect to structuregraph of the original structure.
        """
        ori_checked=Structure.from_str(ori.to(fmt="poscar"),"poscar") 
        opt2_checked=Structure.from_str(opt2.to(fmt="poscar"),"poscar")
        opt_checked=Structure.from_str(opt.to(fmt="poscar"),"poscar")
        std_checked=Structure.from_str(std.to(fmt="poscar"),"poscar")
        sg_ori=self.structure2structure_graph(ori_checked)
        sg_opt=self.structure2structure_graph(opt_checked)
        sg_opt2=self.structure2structure_graph(opt2_checked)
        sg_std=self.structure2structure_graph(std_checked)
        sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, scale=True, attempt_supercell=False, comparator=ElementComparator())
        return str(int(sm.fit(ori_checked, opt2_checked))),str(int(sm.fit(ori_checked, opt_checked))),str(int(sm.fit(ori_checked, std_checked))),str(sg_ori.diff(sg_opt2,strict=False)["dist"]),str(sg_ori.diff(sg_opt,strict=False)["dist"]),str(sg_ori.diff(sg_std,strict=False)["dist"])

    def match_check4(self,ori,opt2,opt,std2,std,ltol=0.2, stol=0.3, angle_tol=5):
        """ (1) Calculate match rates of structure (3), (2), (4) and (1) with respect to the 
                original structure.
            (2) Calculate topological distances of structuregraph of structure (3), (2), (4) 
                and (1) with respect to structuregraph of the original structure.

        Args:
            ori (Structure): Original Structure.
            opt2 (Structure): IAP-optimized Structure.
            opt (Structure): ZL*-Optimized Structure.
            std2 (Structure): IAP-optimized rescaled Structure.
            std (Structure): Rescaled Structure.
            ltol (float, optional): Fractional length tolerance. Default is 0.2.
            stol (float, optional): Site tolerance. Defined as the fraction of the average 
                free length per atom := ( V / Nsites ) ** (1/3). Default is 0.3.
            angle_tol (int, optional): Angle tolerance in degrees. Default is 5.

        Returns:
            str: "1" if IAP-optimized Structure matches original Structure. "0" otherwise.
            str: "1" if ZL*-Optimized Structure matches original Structure. "0" otherwise.
            str: "1" if IAP-optimized rescaled Structure matches original Structure. "0" otherwise.
            str: "1" if Rescaled Structure matches original Structure. "0" otherwise.
            str: The topological distance between IAP-optimized Structure and original Structure.  
            str: The topological distance between ZL*-Optimized Structure and original Structure.  
            str: The topological distance between IAP-optimized Rescaled Structure and original Structure.
            str: The topological distance between Rescaled Structure and original Structure.
        """
        ori_checked=Structure.from_str(ori.to(fmt="poscar"),"poscar") 
        opt2_checked=Structure.from_str(opt2.to(fmt="poscar"),"poscar")
        opt_checked=Structure.from_str(opt.to(fmt="poscar"),"poscar")
        std_checked=Structure.from_str(std.to(fmt="poscar"),"poscar")
        std2_checked=Structure.from_str(std2.to(fmt="poscar"),"poscar")
        sg_ori=self.structure2structure_graph(ori_checked)
        sg_opt=self.structure2structure_graph(opt_checked)
        sg_opt2=self.structure2structure_graph(opt2_checked)
        sg_std=self.structure2structure_graph(std_checked)
        sg_std2=self.structure2structure_graph(std2_checked)
        sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, scale=True, attempt_supercell=False, comparator=ElementComparator())
        return str(int(sm.fit(ori_checked, opt2_checked))),str(int(sm.fit(ori_checked, opt_checked))),str(int(sm.fit(ori_checked, std2_checked))),str(int(sm.fit(ori_checked, std_checked))),str(sg_ori.diff(sg_opt2,strict=False)["dist"]),str(sg_ori.diff(sg_opt,strict=False)["dist"]),str(sg_ori.diff(sg_std2,strict=False)["dist"]),str(sg_ori.diff(sg_std,strict=False)["dist"])
        
