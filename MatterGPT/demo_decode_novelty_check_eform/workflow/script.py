# [File Begins] workflow/script.py
# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
import os,sys,json,gc,math
# One math-library thread per worker process (many workers run in parallel);
# must be set before numpy is imported to take effect.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
from slices.core import SLICES
from pymatgen.core.structure import Structure
import configparser
import time
import os,csv,glob
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.composition import Composition
from contextlib import contextmanager
from functools import wraps
import json
import signal
import time
import argparse
import json 
import sqlite3 
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
import numpy as np

def find_abnormal_lattices(structure: Structure, threshold: float = 10000.0) -> bool: 
    a, b, c = structure.lattice.abc        
    if any(abs(param) > threshold for param in [a, b, c]):
        return True
    return False

def compare_compositions(struct1, struct2):
    return struct1.composition == struct2.composition

ltol = 0.2
stol = 0.3
angle_tol = 5

with open('./chemPotMP.json') as handle: 
    chemPot = json.loads(handle.read())

config = configparser.ConfigParser()
config.read('./settings.ini') 

graph_method = config.get("Settings","graph_method")

check=False
CG=SLICES(graph_method=graph_method, check_results=check)
results=[]

sm = StructureMatcher(ltol, stol, angle_tol, primitive_cell=True, \
    scale=True, attempt_supercell=False, comparator=ElementComparator()) 

DB_PATH = '../structure_database.db'

def process_one_record(prop_values, slices_str):
    global CG
    if not slices_str:
        return
    try:
        if CG.check_SLICES(slices_str, strategy=4, dupli_check=False):
            structure, energy_per_atom = CG.SLICES2structure(slices_str)
            if not find_abnormal_lattices(structure):
                gc.collect()
                finder = SpacegroupAnalyzer(structure)
                try:
                    primitive_standard_structure = finder.get_primitive_standard_structure()
                except Exception as e:
                    print("Warning: get_primitive_standard_structure failed!!!", e)
                    primitive_standard_structure = structure
                try:
                    spacegroup_number = SpacegroupAnalyzer(
                        primitive_standard_structure
                    ).get_space_group_number()
                except Exception as e:
                    print("Warning: get_space_group_number failed!!!", e)
                    spacegroup_number = ""
                comp = primitive_standard_structure.composition

                enthalpyForm = energy_per_atom * comp.num_atoms
                temp = comp.get_el_amt_dict()
                for i in range(len(temp)):
                    enthalpyForm = enthalpyForm - list(temp.values())[i] * chemPot[list(temp.keys())[i]]
                enthalpyForm_per_atom = enthalpyForm / comp.num_atoms

                flag = 0
                new_composition = primitive_standard_structure.composition.reduced_formula

                candidates = []
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("SELECT primitive_cif FROM structures WHERE composition = ?", (new_composition,))
                    candidates = c.fetchall()
                    conn.close()
                except sqlite3.Error as e:
                    print(f"Database query failed: {e}")
                for (db_cif_string,) in candidates:
                    db_stru = Structure.from_str(db_cif_string, "cif")
                    if sm.fit(db_stru, primitive_standard_structure):
                        flag = 1
                        break
                novelty = 0 if flag else 1

                with open("result2.csv", 'a') as fn:
                    fn.write(
                        ",".join(prop_values) + "," + slices_str + "," + str(enthalpyForm_per_atom)
                        + "," + str(spacegroup_number) + ',"' + primitive_standard_structure.to(fmt="poscar")
                        + '",' + str(novelty) + "\n"
                    )
    except Exception as e:
        del CG
        CG = SLICES(graph_method=graph_method, check_results=check)
        print(e)
        gc.collect()


with open('temp_splited.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

if rows and (("SLICES" in rows[0]) or ("symSLICES" in rows[0])):
    header = [h.strip() for h in rows[0]]
    data_rows = rows[1:]
    slices_col = "SLICES" if "SLICES" in header else "symSLICES"
    prop_cols = [h for h in header if h not in {slices_col, "crystal_system"}]
    if not prop_cols and header:
        prop_cols = [header[0]]

    for row in data_rows:
        if not row:
            continue
        row_map = {h: row[i].strip() if i < len(row) else "" for i, h in enumerate(header)}
        prop_values = [row_map.get(c, "") for c in prop_cols]
        slices_str = row_map.get(slices_col, "").strip()
        process_one_record(prop_values, slices_str)
else:
    for row in rows:
        if not row:
            continue
        slices_str = row[-1].strip()
        prop_values = [row[0].strip()] if row else [""]
        process_one_record(prop_values, slices_str)
            
