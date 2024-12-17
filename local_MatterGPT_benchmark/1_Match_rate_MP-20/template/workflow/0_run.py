# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
import sys
import json
import gc
from slices.core import SLICES
from pymatgen.core.structure import Structure
import configparser
import time

os.environ["OMP_NUM_THREADS"] = "1"

config = configparser.ConfigParser()
config.read('../settings.ini')  # path of your .ini file
bond_scaling = config.getfloat("Settings", "bond_scaling")
delta_theta = config.getfloat("Settings", "delta_theta")
delta_x = config.getfloat("Settings", "delta_x")
lattice_shrink = config.getfloat("Settings", "lattice_shrink")
lattice_expand = config.getfloat("Settings", "lattice_expand")
angle_weight = config.getfloat("Settings", "angle_weight")
vbond_param_ave_covered = config.getfloat("Settings", "vbond_param_ave_covered")
vbond_param_ave = config.getfloat("Settings", "vbond_param_ave")
repul = config.getboolean("Settings", "repul")
graph_method = config.get("Settings", "graph_method")
print(delta_theta, lattice_expand)

# This loads the default pre-trained model
with open('temp.json', 'r') as f:
    cifs = json.load(f)

# Remove existing result.csv to prevent conflicts (slurm's twice execution bug)
if os.path.exists("result.csv"):
    os.remove("result.csv")

check = False
CG = SLICES(graph_method=graph_method, check_results=check)

# 初始化一个列表来存储所有结果
result_lines = []



for i in range(len(cifs)):
    p = cifs[i]["cif"]  # path to CIF file
    try:
        start_time = time.time()
        ori = Structure.from_str(p, "cif")
        num_atoms = len(ori.atomic_numbers)
        CG.from_cif(p)
        # print(bond_scaling, delta_theta, delta_x,lattice_shrink,lattice_expand,angle_weight,epsilon,repul)
        structures, energy = CG.to_structures(
            bond_scaling,
            delta_theta,
            delta_x,
            lattice_shrink,
            lattice_expand,
            angle_weight,
            vbond_param_ave_covered,
            vbond_param_ave,
            repul
        )
        if len(structures) == 3:
            a, b, c, d, e, f = CG.match_check3(ori, structures[2], structures[1], structures[0])
            a2, b2, c2, d2, e2, f2 = CG.match_check3(
                ori, structures[2], structures[1], structures[0], ltol=0.3, stol=0.5, angle_tol=10
            )
            time_used = time.time() - start_time
            line = f"{cifs[i]['material_id']},{a},{b},{c},{a2},{b2},{c2},{d2},{e2},{f2},{num_atoms},{time_used}\n"
            result_lines.append(line)
        elif len(structures) == 2:
            a, b, c, d = CG.match_check(ori, structures[1], structures[0])
            a2, b2, c2, d2 = CG.match_check(
                ori, structures[1], structures[0], ltol=0.3, stol=0.5, angle_tol=10
            )
            time_used = time.time() - start_time
            line = f"{cifs[i]['material_id']},0,{a},{b},0,{a2},{b2},1,{c2},{d2},{num_atoms},{time_used}\n"
            result_lines.append(line)
    except Exception as e1:
        # Reinitialize CG in case of an exception
        del CG
        gc.collect()  # Ensure garbage collection
        CG = SLICES(graph_method=graph_method, check_results=check)
        # Record the exception message
        error_message = str(e1).split('\n')[0]
        line = f"{cifs[i]['material_id']},{error_message}\n"
        result_lines.append(line)

# 一次性写入所有结果到 'result.csv'
with open("result.csv", 'w') as fn:
    fn.writelines(result_lines)

print("Processing complete. Results have been written to result.csv.")
