# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
import json
from slices.utils import splitRun,show_progress,collect_json,collect_csv
from itertools import zip_longest
# An optional utility to display a progress bar
# for long-running loops. `pip install tqdm`.
from tqdm import tqdm
from pymatgen.ext.matproj import MPRester
import pandas as pd

output=[]
data=pd.read_csv("../../../data/mp20/test.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})
data=pd.read_csv("../../../data/mp20/val.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})
data=pd.read_csv("../../../data/mp20/train.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i]})
with open('cifs.json', 'w') as f:
    json.dump(output, f)
splitRun(filename='./cifs.json',threads=16,skip_header=False)

show_progress()
collect_json(output="cifs_filtered.json", \
    glob_target="./**/output.json",cleanup=True)
