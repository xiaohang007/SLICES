# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
from slices.utils import splitRun,show_progress,collect_json,collect_csv
import json
import pandas as pd
output=[]
data_path_predix="../mp20/"
data=pd.read_csv(data_path_predix+"test.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
bandgap=list(data["band_gap"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i],"band_gap":bandgap[i]})
data=pd.read_csv(data_path_predix+"val.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
bandgap=list(data["band_gap"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i],"band_gap":bandgap[i]})
data=pd.read_csv(data_path_predix+"train.csv")
cifs=list(data["cif"])
ids=list(data["material_id"])
eform=list(data["formation_energy_per_atom"])
bandgap=list(data["band_gap"])
for i in range(len(ids)):
    output.append({"material_id":ids[i],"formation_energy_per_atom":eform[i],"cif":cifs[i],"band_gap":bandgap[i]})
with open('cifs.json', 'w') as f:
    json.dump(output, f)

splitRun(filename='./cifs.json',threads=16,skip_header=False)
show_progress()
collect_json(output="cifs_filtered.json", \
    glob_target="./**/output.json",cleanup=False)
collect_csv(output="mp20_eform_bandgap_nonmetal.csv", \
    glob_target="./**/result.csv",cleanup=True,header="SLICES,eform,bandgap\n")
os.system("rm cifs.json")