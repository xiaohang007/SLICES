# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
"""
Created on Wed Dec 19 18:47:34 2018
读取构型文件，自动计算生成焓
@author: xiaoh
"""
import os
import json
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPRelaxSet,MPStaticSet
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from jarvis.io.vasp.inputs import Poscar, Incar, Potcar
from jarvis.core.kpoints import Kpoints3D as Kpoints
s = Structure.from_file("./EFormation/CONTCAR")
mat = Poscar.from_file("./EFormation/CONTCAR")
scf = MPStaticSet(s,user_incar_settings={'NCORE': 8,'ISMEAR':0,
          "NELM": 500,
            "LORBIT": 11,
            "LCHARG": ".TRUE.",
            "LWAVE": ".FALSE.","LDAU": ".FALSE."},force_gamma=True)
scf.write_input("Scf")
os.system("cp ./EFormation/CHGCAR ./Scf/")
