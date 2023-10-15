# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
"""
Created on Wed Dec 19 18:47:34 2018
读取构型文件，自动计算生成焓
@author: xiaoh
"""
import os
import json
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.compatibility import MaterialsProjectCompatibility

s = Structure.from_file("./EFormation0.5/CONTCAR")
relax = MPRelaxSet(s,user_incar_settings={'NCORE': 8,"ISYM":0},force_gamma=True)
relax.write_input("EFormation")
os.chdir('EFormation')
