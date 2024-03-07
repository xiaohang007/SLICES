# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import json
from invcryrep.utils import temporaryWorkingDirectory,search_materials,exclude_elements_json,splitRun,show_progress,collect_json,collect_csv
# Download entries to build general and transfer datasets
dict_json=search_materials(apikeyPath='/crystal/APIKEY.ini',formation_energy=(-10000,0),num_sites=(1,5),fields=["material_id"])
exclude_elements=['Fr' , 'Ra','Ac','Th','Pa','U','Np',\
            'Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf',\
            'Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc',\
            'Lv','Ts','Og']
flitered_json=exclude_elements_json(dict_json,exclude_elements)
with open('prior_model_dataset.json', 'w') as f:
    json.dump(flitered_json, f)
print("prior_model_dataset.json generated")
dict_json2=search_materials(apikeyPath='/crystal/APIKEY.ini',band_gap=(0.10, 0.55),num_sites=(1,5),formation_energy=(-10000,0),is_gap_direct=True,fields=["material_id"])
flitered_json2=exclude_elements_json(dict_json2,exclude_elements)
with open('transfer_learning_dataset.json', 'w') as f:
    json.dump(flitered_json2, f)
print("transfer_learning_dataset.json generated")
# Rule out crystals with low-dimensional units (e.g. molecular crystals or layered crystals)
with temporaryWorkingDirectory("./2_filter_prior_3d"):
    splitRun(filename='../prior_model_dataset.json',threads=8,skip_header=False)
    show_progress()
    collect_json(output="../prior_model_dataset_filtered.json", glob_target="./**/output.json",cleanup=True)
with temporaryWorkingDirectory("./3_filter_transfer_3d"):
    splitRun(filename='../transfer_learning_dataset.json',threads=8,skip_header=False)
    show_progress()
    collect_json(output="../transfer_model_dataset_filtered.json", glob_target="./**/output.json",cleanup=True)