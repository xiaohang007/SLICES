# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os
import re
import csv
import sys
import time
import argparse
import pickle,json
from monty.serialization import loadfn, dumpfn
from tqdm import tqdm
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN,BrunnerNN_reciprocal,EconNN,MinimumDistanceNN
import numpy as np
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
ssf = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
    stats=('mean', 'std_dev', 'minimum', 'maximum'))

fingerprint_list=[]

def main(args):
    structure_graph_list=[]
    with open(args.input, 'r') as f:
        cifs=json.load(f)
    for i  in range(len(cifs)):
        cif_string=cifs[i]["cif"]
        structure = Structure.from_str(cif_string,"cif")
        fingerprint_list.append(np.array(ssf.featurize(structure)))
        print(i)
    dumpfn(fingerprint_list, "./training_fingerprint_list.json.gz") # or .json.gz






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculates energy above the convex hull for a csv of formulas and total energies (with header)')
    parser.add_argument('--debug', action='store_false',
                        help='Enable debug mode (test imports)')
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Input csv filename')
    parser.add_argument('-o', '--output', type=str,
                        required=True, help='Output csv filename')
    args = parser.parse_args(sys.argv[1:])

    assert os.path.exists(args.input), 'Input file does not exist'

    main(args)
