# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json
from slices.core import SLICES
from pymatgen.core.structure import Structure
import configparser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"

config = configparser.ConfigParser()
config.read('../settings.ini') #path of your .ini file
bond_scaling = config.getfloat("Settings","bond_scaling") 
delta_theta = config.getfloat("Settings","delta_theta") 
delta_x = config.getfloat("Settings","delta_x") 
lattice_shrink = config.getfloat("Settings","lattice_shrink") 
lattice_expand = config.getfloat("Settings","lattice_expand") 
angle_weight = config.getfloat("Settings","angle_weight") 
epsilon = config.getfloat("Settings","epsilon") 
repul = config.getboolean("Settings","repul") 
graph_method = config.get("Settings","graph_method")
print(delta_theta,lattice_expand)
with open('temp.json', 'r') as f:
    cifs=json.load(f)
cifs_filtered=[]
CG=SLICES(graph_method=graph_method, check_results=False)
for i  in range(len(cifs)):
    cif_string=cifs[i]["cif"]
    if len(sys.argv)==2:
        if sys.argv[1]=="test":
            check=True
    else:
        check=False
    if check:
        ori = Structure.from_str(cif_string,"cif")
        dim=CG.get_dim(ori)
        if dim==3:
            cifs_filtered.append(cifs[i])
        with open("result.csv",'a') as f:
            f.write(cifs[i]["qmof_id"]+','+str(dim)+'\n')  
    else:
        try:
            ori = Structure.from_str(cif_string,"cif")
            num_ori=len(np.array(ori.atomic_numbers))
            sga = SpacegroupAnalyzer(ori)
            ori_pri = sga.get_primitive_standard_structure()
            num_ori_pri=len(np.array(ori_pri.atomic_numbers))
            ori_pri.to(cifs[i]["qmof_id"]+'.cif','cif')
            with open(cifs[i]["qmof_id"]+'.cif') as f:
                temp=f.read()
            os.system("rm "+cifs[i]["qmof_id"]+'.cif')
            if num_ori_pri < num_ori:
                print(cifs[i]["qmof_id"]+" : primitive")
                cifs[i]["cif"]=temp
                cifs_filtered.append(cifs[i])
            else:
                cifs_filtered.append(cifs[i])
        except Exception as e1:
            with open("result.csv",'a') as f2:
                f2.write(cifs[i]["qmof_id"]+','+str(e1)+'\n')

with open('output.json', 'w') as f:
    json.dump(cifs_filtered, f)



