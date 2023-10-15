# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import json
from itertools import zip_longest

# An optional utility to display a progress bar
# for long-running loops. `pip install tqdm`.
from tqdm import tqdm
import json
from pymatgen.ext.matproj import MPRester
import os
import pandas as pd

output=[]
data=pd.read_csv("test.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})
data=pd.read_csv("val.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})
data=pd.read_csv("train.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})

with open('cifs.json', 'w') as f:
    json.dump(output, f)
