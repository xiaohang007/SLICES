# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json
from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
import configparser
os.environ["OMP_NUM_THREADS"] = "1"

config = configparser.ConfigParser()
config.read('./settings.ini') #path of your .ini file

graph_method = config.get("Settings","graph_method")
augment = config.getfloat("Settings","augment")

with open('temp.json', 'r') as f:
    cifs=json.load(f)
cifs_filtered=[]
sci_text=''
CG=InvCryRep(graph_method=graph_method)
for i  in range(len(cifs)):
    cif_string=cifs[i]["cif"]
    eform=cifs[i]["formation_energy_per_atom"]

    ori = Structure.from_str(cif_string,"cif")
    sci_list=CG.structure2SLICESAug(ori,3,augment)

    with open("result.csv",'a') as f:
        for j in sci_list:
            sci_text+=j+','+str(eform)+'\n'
            f.write(cifs[i]["material_id"]+','+j+'\n') 


with open("result.sli","w") as f:
    f.write(sci_text)



