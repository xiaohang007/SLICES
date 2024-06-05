# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json,gc,math
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
import configparser
import time
import os,csv,glob
from jarvis.io.vasp.outputs import Outcar, Vasprun
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.composition import Composition
from contextlib import contextmanager
from functools import wraps
import json
import signal
os.environ["OMP_NUM_THREADS"] = "1"


with open('./chemPotMP.json') as handle:
    chemPot = json.loads(handle.read())

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
CG=InvCryRep(graph_method=graph_method, check_results=check)
results=[]
with open('temp_splited.csv', 'r') as f:
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
                comp = primitive_standard_structure.composition
                enthalpyForm=energy_per_atom*comp.num_atoms
                temp=comp.get_el_amt_dict()
                for i in range(len(temp)):
                    enthalpyForm=enthalpyForm-list(temp.values())[i]*chemPot[list(temp.keys())[i]]
                enthalpyForm_per_atom=enthalpyForm/comp.num_atoms

                with open("result.csv",'a') as fn:
                    fn.write(row[0]+','+row[1]+','+str(enthalpyForm_per_atom)+','+primitive_standard_structure.to(fmt="poscar").replace('\n','\\n')+'\n')
        except Exception as e:
            print(e)


