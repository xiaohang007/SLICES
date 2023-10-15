# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json,gc,math
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
import configparser
import time


config = configparser.ConfigParser()
config.read('../settings.ini') #path of your .ini file
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
with open('temp_splited.sci', 'r') as f:
    sciles_list=f.readlines()

for i  in range(len(sciles_list)):
    sciles=sciles_list[i].strip()
    check=False
    try:
        CG.from_SLICES(sciles)
        #print(bond_scaling, delta_theta, delta_x,lattice_shrink,lattice_expand,angle_weight,epsilon,repul)
        structure,energy_per_atom=CG.to_relaxed_structure(bond_scaling, delta_theta, delta_x, \
        lattice_shrink,lattice_expand,angle_weight,vbond_param_ave_covered,vbond_param_ave,repul)
        if CG.check_structural_validity(structure):
            with open("result.csv",'a') as fn:
                fn.write(sciles+','+str(structure.formula).replace(' ','')+','+str(energy_per_atom)+ \
                ','+structure.to(fmt="poscar").replace('\n','\\n')+'\n')
    except:
        pass


