# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
"""
Created on Wed Dec 19 18:47:34 2018
读取构型文件，自动计算生成焓
@author: xiaoh
"""
import re
import os
import json
from pymatgen.io.vasp.inputs import Kpoints, Incar
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#读取化学势数据库

#准备能量计算文件夹
s = Structure.from_file("temp.cif")
finder = SpacegroupAnalyzer(s)
try:
    primitive_standard_structure = finder.get_primitive_standard_structure()
except Exception as e:
    print("Error: get_primitive_standard_structure failed!!!",e)
    primitive_standard_structure = s
relax = MPRelaxSet(primitive_standard_structure,user_incar_settings={'NCORE': 9,'EDIFF': 0.0001,'EDIFFG': -0.05,'LREAL':".FALSE.",  \
'NSW': 200,'ALGO': 'Normal','PREC': 'Normal','ISMEAR': -5,'ISIF': 3,'LORBIT':'.FALSE.','LCHARG':'.FALSE.',"LWAVE": ".FALSE.","LDAU":".FALSE.","ISPIN":1,"ISYM":0}, \
user_kpoints_settings={'reciprocal_density': 32},force_gamma=True)
relax.write_input("EFormation0")
os.chdir('EFormation0')
