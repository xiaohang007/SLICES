# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com
import os
import csv
import gc
import json
import tensorflow as tf
import configparser
from slices.core import SLICES
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"


def find_abnormal_lattices(structure: Structure, threshold: float = 10000.0) -> bool:
    a, b, c = structure.lattice.abc
    return any(abs(param) > threshold for param in [a, b, c])


# Load chemical potentials (same logic as demo)
chem_pot_path = "./chemPotMP.json"
if not os.path.exists(chem_pot_path):
    alt_path = "../demo_decode_novelty_check_eform_m3gnet/workflow/chemPotMP.json"
    if os.path.exists(alt_path):
        chem_pot_path = alt_path
    else:
        raise FileNotFoundError(
            "chemPotMP.json not found. Place it in 2_decode/workflow or use the demo copy."
        )

with open(chem_pot_path, "r", encoding="utf-8") as handle:
    chem_pot = json.loads(handle.read())

# 读取配置文件以获取 graph_method
config = configparser.ConfigParser()
config.read("./settings.ini")
graph_method = config.get("Settings", "graph_method", fallback="default")

# Initialize SLICES
check = False
CG = SLICES(graph_method=graph_method, check_results=check, relax_model="m3gnet")

with open("temp_splited.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        try:
            slices_str = row[-1].strip()
            if CG.check_SLICES(slices_str, strategy=4, dupli_check=False):
                structure, energy_per_atom = CG.SLICES2structure(slices_str)
                if not find_abnormal_lattices(structure):
                    finder = SpacegroupAnalyzer(structure)
                    try:
                        primitive_structure = finder.get_primitive_standard_structure()
                    except Exception:
                        primitive_structure = structure

                    try:
                        spacegroup_number = SpacegroupAnalyzer(
                            primitive_structure
                        ).get_space_group_number()
                    except Exception:
                        spacegroup_number = ""

                    comp = primitive_structure.composition
                    enthalpy_form = energy_per_atom * comp.num_atoms
                    for elem, amt in comp.get_el_amt_dict().items():
                        enthalpy_form -= amt * chem_pot[elem]
                    enthalpy_form_per_atom = enthalpy_form / comp.num_atoms

                    poscar_str = primitive_structure.to(fmt="poscar").replace("\n", "\\n")

                    with open("result2.csv", "a", encoding="utf-8", newline="") as fn:
                        writer = csv.writer(fn)
                        writer.writerow(
                            row
                            + [
                                str(enthalpy_form_per_atom),
                                str(spacegroup_number),
                                poscar_str,
                            ]
                        )
        except Exception as e:
            print(f"Error: {e}")
            del CG
            CG = SLICES(graph_method=graph_method, check_results=check, relax_model="m3gnet")
            gc.collect()
            tf.keras.backend.clear_session()
