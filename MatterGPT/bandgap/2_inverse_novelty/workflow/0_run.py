# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
import os,sys,json,gc,math
from slices.core import SLICES
from pymatgen.core.structure import Structure
import configparser
import time
import os,csv,glob
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.composition import Composition
from contextlib import contextmanager
from functools import wraps
import json
import signal
import time
import argparse
import pickle,json
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"

def compare_compositions(struct1, struct2):
    return struct1.composition == struct2.composition

ltol = 0.2
stol = 0.3
angle_tol = 5
with open('./chemPotMP.json') as handle:
    chemPot = json.loads(handle.read())

with open('../structure_database.pkl', 'rb') as f:
    structure_database = pickle.load(f)
print(structure_database[0])
config = configparser.ConfigParser()
config.read('./settings.ini') #path of your .ini file
bond_scaling = config.getfloat("Settings","bond_scaling") 
delta_theta = config.getfloat("Settings","delta_theta") 
delta_x = config.getfloat("Settings","delta_x") 
lattice_shrink = config.getfloat("Settings","lattice_shrink") 
lattice_expand = config.getfloat("Settings","lattice_expand") 
angle_weight = config.getfloat("Settings","angle_weight") 
vbond_param_ave_covered = config.getfloat("Settings","vbond_param_ave_covered") 
vbond_param_ave = config.getfloat("Settings","vbond_param_ave") 
repul = config.getboolean("Settings","repul") 
graph_method = config.get("Settings","graph_method")

check=False
CG=SLICES(graph_method=graph_method, check_results=check, relax_model="m3gnet")
results=[]
with open('temp.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            if CG.check_SLICES(row[1],strategy=4,dupli_check=False):
                CG.from_SLICES(row[1],strategy=4,fix_duplicate_edge=True)
                #print(bond_scaling, delta_theta, delta_x,lattice_shrink,lattice_expand,angle_weight,epsilon,repul)
                structure,energy_per_atom=CG.to_relaxed_structure(bond_scaling, delta_theta, delta_x, \
                lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul)
                finder = SpacegroupAnalyzer(structure)
                try:
                    primitive_standard_structure = finder.get_primitive_standard_structure()
                except Exception as e:
                    print("Warning: get_primitive_standard_structure failed!!!",e)
                    primitive_standard_structure = structure
       
                flag = 0  # dissimilar
                for struc in structure_database:
                    if compare_compositions(struc[0], primitive_standard_structure):
                        sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, \
                        scale=True, attempt_supercell=False, comparator=ElementComparator())
                        if sm.fit(struc[0],primitive_standard_structure):
                            flag = 1
                            continue             
                if flag:        
                    novelty=0
                else:
                    novelty=1
                with open("result.csv",'a') as fn:
                    fn.write(row[0]+','+row[1] +','+primitive_standard_structure.to(fmt="poscar").replace('\n','\\n')+','+str(novelty)+'\n')
        except Exception as e:
            del CG
            CG=SLICES(graph_method=graph_method, check_results=check, relax_model="m3gnet")
            print(e)



