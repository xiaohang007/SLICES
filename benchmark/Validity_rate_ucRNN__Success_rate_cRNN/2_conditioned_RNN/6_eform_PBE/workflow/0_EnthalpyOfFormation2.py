# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:47:34 2018
读取构型文件，自动计算生成焓
@author: xiaoh
"""
import json
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
#from pymatgen.ext.matproj import MPRester
#读取化学势数据库
#a = MPRester("PWvnIjizov7K5a3nLw6y")


with open('./chemPotMP.json') as handle:
    chemPot = json.loads(handle.read())
#读取计算结果
drone = VaspToComputedEntryDrone()
queen = BorgQueen(drone, "EFormation", 1)
entries = queen.get_data()
#修正计算结果并计算生成焓
compat = MaterialsProjectCompatibility()
compat.process_entries(entries)
entry = entries[0]
enthalpyForm=entry.energy
temp=entry.composition.get_el_amt_dict()
for i in range(len(temp)):
    enthalpyForm=enthalpyForm-list(temp.values())[i]*chemPot[list(temp.keys())[i]]
enthalpyForm=enthalpyForm/entry.composition.num_atoms
f = open("results.txt", "w")
f.write(str(enthalpyForm))

