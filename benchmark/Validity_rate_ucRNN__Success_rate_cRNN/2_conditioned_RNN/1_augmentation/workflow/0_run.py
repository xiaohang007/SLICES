# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
import os,sys,json
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
import configparser
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
sci_text=''
CG=InvCryRep(graph_method=graph_method)
for i  in range(len(cifs)):
    cif_string=cifs[i]["cif"]
    eform=cifs[i]["formation_energy_per_atom"]
    try:
        ori = Structure.from_str(cif_string,"cif")
        sci_list=CG.structure2SLICESAug(ori,3,50)

        with open("result.csv",'a') as f:
            for j in sci_list:
                sci_text+=j+','+str(eform)+'\n'
                f.write(cifs[i]["material_id"]+','+j+'\n') 
    except Exception as e1:
        with open("result.csv",'a') as f2:
            f2.write(cifs[i]["material_id"]+',error\n')

with open("result.sci","w") as f:
    f.write(sci_text)


