# -*- coding: utf-8 -*-
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
for i  in range(len(cifs)):
    cif_string=cifs[i]["cif"]
    if len(sys.argv)==2:
        if sys.argv[1]=="test":
            check=True
    else:
        check=False
    if check:
        CG=InvCryRep(graph_method=graph_method, check_results=check)
        CG.from_cif(cif_string)
        if CG.check_element():
            cifs_filtered.append(cifs[i])
        with open("result.csv",'a') as f:
            f.write(cifs[i]["material_id"]+'\n')  
    else:
        try:
            CG=InvCryRep(graph_method=graph_method, check_results=check)
            CG.from_cif(cif_string)
            if CG.check_element():
                cifs_filtered.append(cifs[i])
            with open("result.csv",'a') as f:
                f.write(cifs[i]["material_id"]+'\n')  
        except Exception as e1:
            with open("result.csv",'a') as f2:
                f2.write(cifs[i]["material_id"]+','+str(e1)+'\n')

with open('output.json', 'w') as f:
    json.dump(cifs_filtered, f)



