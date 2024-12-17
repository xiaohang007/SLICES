# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
import sys
import json
from slices.core import SLICES
from pymatgen.core.structure import Structure
import configparser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

# 读取配置文件
config = configparser.ConfigParser()
config.read('./settings.ini')  # .ini 文件路径
bond_scaling = config.getfloat("Settings", "bond_scaling") 
delta_theta = config.getfloat("Settings", "delta_theta") 
delta_x = config.getfloat("Settings", "delta_x") 
lattice_shrink = config.getfloat("Settings", "lattice_shrink") 
lattice_expand = config.getfloat("Settings", "lattice_expand") 
angle_weight = config.getfloat("Settings", "angle_weight") 
epsilon = config.getfloat("Settings", "epsilon") 
repul = config.getboolean("Settings", "repul") 
graph_method = config.get("Settings", "graph_method")

print(delta_theta, lattice_expand)

# 读取 CIF 数据
with open('temp.json', 'r') as f:
    cifs = json.load(f)

cifs_filtered = []
CG = SLICES(graph_method=graph_method, relax_model="chgnet")

# 用于存储 CSV 行的数据
csv_lines = []

for cif in cifs:
    cif_string = cif["cif"]
    try:
        ori = Structure.from_str(cif_string, "cif")
        if CG.check_element(ori):
            if cif["band_gap"] > 0.01:
                if CG.check_3D(ori):
                    num_ori = len(ori.atomic_numbers)
                    sga = SpacegroupAnalyzer(ori)
                    ori_pri = sga.get_primitive_standard_structure()
                    num_ori_pri = len(ori_pri.atomic_numbers)
                    
                    # 保存原始结构为 CIF 文件
                    cif_filename = f"{cif['material_id']}.cif"
                    ori_pri.to(cif_filename, 'cif')
                    
                    with open(cif_filename, 'r') as f_cif:
                        temp_cif = f_cif.read()
                    
                    # 删除临时 CIF 文件
                    os.remove(cif_filename)
                    
                    if num_ori_pri < num_ori:
                        cif["cif"] = temp_cif
                        cifs_filtered.append(cif)
                        csv_line = f"{CG.structure2SLICES(ori_pri)},{cif['formation_energy_per_atom']},{cif['band_gap']}\n"
                        csv_lines.append(csv_line)
                    else:
                        cifs_filtered.append(cif)
                        csv_line = f"{CG.structure2SLICES(ori)},{cif['formation_energy_per_atom']},{cif['band_gap']}\n"
                        csv_lines.append(csv_line)
    except Exception as e1:
        print(e1)

# 将过滤后的 CIF 数据保存到 output.json
with open('output.json', 'w') as f_out:
    json.dump(cifs_filtered, f_out)

# 在循环结束后一次性写入 result.csv
with open("result.csv", 'w') as f_csv:
    # 如果需要添加表头，可以取消下面一行的注释
    # f_csv.write("SLICES,Formation Energy per Atom,Band Gap\n")
    f_csv.writelines(csv_lines)

print("处理完成，结果已保存到 output.json 和 result.csv")

