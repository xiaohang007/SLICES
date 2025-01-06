# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
import os,sys,json,gc,math
from slices.core import SLICES
from pymatgen.core.structure import Structure
import configparser
import time
import os,csv,glob
import tensorflow as tf
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
def find_abnormal_lattices(structure: Structure, threshold: float = 10000.0) -> bool:
    """
    检查给定 Structure 对象的晶格向量中是否有任意分量的绝对值大于阈值。
    
    参数:
    - structure (Structure): 要检查的 pymatgen Structure 对象。
    - threshold (float): 判断异常的阈值，默认值为 10,000.0。
    
    返回:
    - bool: 如果任意晶格参数的绝对值超过阈值，则返回 True，否则返回 False。
    """
    a, b, c = structure.lattice.abc  # 获取晶格向量的长度
    # 检查 a, b, c 是否有任意一个的绝对值超过阈值
    if any(abs(param) > threshold for param in [a, b, c]):
        return True
    return False

def compare_compositions(struct1, struct2):
    return struct1.composition == struct2.composition

ltol = 0.2
stol = 0.3
angle_tol = 5

with open('./chemPotMP.json') as handle:
    chemPot = json.loads(handle.read())
# Load the serialized data from the parent directory
with open('../structure_database.pkl', 'rb') as f:
    structure_database = pickle.load(f)
print(structure_database[0])
config = configparser.ConfigParser()
config.read('./settings.ini') #path of your .ini file

graph_method = config.get("Settings","graph_method")

check=False
CG=SLICES(graph_method=graph_method, check_results=check, relax_model="m3gnet")
results=[]
with open('temp_splited.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            if CG.check_SLICES(row[1],strategy=4,dupli_check=False):
                #print(bond_scaling, delta_theta, delta_x,lattice_shrink,lattice_expand,angle_weight,epsilon,repul)
                structure,energy_per_atom=CG.SLICES2structure(row[1])
                if not find_abnormal_lattices(structure):
                    gc.collect()
                    tf.keras.backend.clear_session()
                    finder = SpacegroupAnalyzer(structure)
                    try:
                        primitive_standard_structure = finder.get_primitive_standard_structure()
                    except Exception as e:
                        print("Warning: get_primitive_standard_structure failed!!!",e)
                        primitive_standard_structure = structure
                    comp = primitive_standard_structure.composition
                    enthalpyForm=energy_per_atom*comp.num_atoms
                    temp=comp.get_el_amt_dict()
                    for i in range(len(temp)):
                        enthalpyForm=enthalpyForm-list(temp.values())[i]*chemPot[list(temp.keys())[i]]
                    enthalpyForm_per_atom=enthalpyForm/comp.num_atoms
           
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
                    with open("result2.csv",'a') as fn:
                        fn.write(row[0]+','+row[1]+','+str(enthalpyForm_per_atom) \
                        +',"'+primitive_standard_structure.to(fmt="poscar")+'",'+str(novelty)+'\n')
        except Exception as e:
            del CG
            CG=SLICES(graph_method=graph_method, check_results=check, relax_model="m3gnet")
            print(e)
            gc.collect()
            tf.keras.backend.clear_session()


