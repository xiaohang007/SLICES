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
graph_method = config.get("Settings", "graph_method")

# 读取 CIF 数据
with open('temp.json', 'r') as f:
    cifs = json.load(f)

# 获取第一个样本的所有属性键（除了'cif'）
if cifs:
    property_keys = [key for key in cifs[0].keys() if key != 'cif']
else:
    print("Error: No data found in temp.json")
    sys.exit(1)

cifs_filtered = []
CG = SLICES(graph_method=graph_method, relax_model="chgnet")
csv_lines = []

for cif in cifs:
    cif_string = cif["cif"]
    try:
        ori = Structure.from_str(cif_string, "cif")
        if CG.get_dim(ori) == 3:
            num_ori = len(ori.atomic_numbers)
            sga = SpacegroupAnalyzer(ori)
            crystal_system = sga.get_crystal_system().lower()  # 获取晶系信息
            ori_pri = sga.get_primitive_standard_structure()
            num_ori_pri = len(ori_pri.atomic_numbers)
            
            # 保存原始结构为 CIF 文件
            ori_pri.to("temp.cif", 'cif')
            with open("temp.cif", 'r') as f_cif:
                temp_cif = f_cif.read()
            os.remove("temp.cif")
            
            if num_ori_pri < num_ori:
                cif["cif"] = temp_cif
                cifs_filtered.append(cif)
                # 动态构建CSV行，添加晶系信息
                property_values = [str(cif.get(key, '')) for key in property_keys]
                csv_line = f"{CG.structure2SLICES(ori_pri)},{','.join(property_values)},{crystal_system}\n"
                csv_lines.append(csv_line)
            else:
                ori.to("temp.cif", 'cif')
                with open("temp.cif", 'r') as f_cif:
                    temp_cif = f_cif.read()
                os.remove("temp.cif")
                cif["cif"] = temp_cif
                cifs_filtered.append(cif)
                # 动态构建CSV行，添加晶系信息
                property_values = [str(cif.get(key, '')) for key in property_keys]
                csv_line = f"{CG.structure2SLICES(ori)},{','.join(property_values)},{crystal_system}\n"
                csv_lines.append(csv_line)
    except Exception as e1:
        print(f"Error processing CIF for material_id {cif.get('material_id', 'N/A')}: {e1}")

# 将过滤后的 CIF 数据保存到 output.json
with open('output.json', 'w') as f_out:
    json.dump(cifs_filtered, f_out)

# 写入CSV文件，包含表头
with open("result.csv", 'w') as f_csv:
    f_csv.writelines(csv_lines)

print("处理完成，结果已保存到 output.json 和 result.csv")
