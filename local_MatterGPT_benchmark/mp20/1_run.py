# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
from utils import *
import json
import pandas as pd
output=[]
data_path_predix="./"
data=pd.read_csv(data_path_predix+"test.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})
data=pd.read_csv(data_path_predix+"val.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})
data=pd.read_csv(data_path_predix+"train.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})
with open('cifs.json', 'w') as f:
    json.dump(output, f)

splitRun_local(filename='./cifs.json',threads=8,skip_header=False)
show_progress_local()
collect_json(output="cifs_filtered.json", \
    glob_target="./**/output.json",cleanup=True)