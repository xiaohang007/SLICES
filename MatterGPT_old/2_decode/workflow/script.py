# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
import os
import csv
import gc
import tensorflow as tf
from slices.core import SLICES
from pymatgen.core.structure import Structure
import configparser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from m3gnet.models import Relaxer

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"

# 读取配置文件
config = configparser.ConfigParser()
config.read('./settings.ini')
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

# 检查晶格是否异常的函数
def find_abnormal_lattices(structure: Structure, threshold: float = 10000.0) -> bool:
    a, b, c = structure.lattice.abc
    if any(abs(param) > threshold for param in [a, b, c]):
        return True
    return False

# 初始化 SLICES 和 Relaxer
check = False
CG = SLICES(graph_method=graph_method, check_results=check, relax_model="m3gnet")
relaxer = Relaxer(optimizer="FIRE")

# 处理 CSV 文件
with open('temp_splited.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        try:
            slices = row[-1].strip()
            if CG.check_SLICES(slices, strategy=4, dupli_check=False):
                # 生成结构
                structures, energy = CG.to_structures(
                    bond_scaling, delta_theta, delta_x,
                    lattice_shrink, lattice_expand, angle_weight,
                    vbond_param_ave_covered, vbond_param_ave, repul
                )
                structure = structures[-1]
                if not find_abnormal_lattices(structure):
                    # 获取精修结构
                    finder = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=15)
                    refined_structure = finder.get_refined_structure()
                    final_space_group = finder.get_space_group_number()

                    # 输出结果到文件
                    with open("result2.csv", 'a') as fn:
                        result_row = row + [ str(final_space_group),refined_structure.to(fmt="poscar").replace('\n','\\n')]
                        fn.write(",".join(result_row) + '\n')
        except Exception as e:
            print(f"Error: {e}")
            del CG
            CG = SLICES(graph_method=graph_method, check_results=check, relax_model="m3gnet")
            gc.collect()
            tf.keras.backend.clear_session()
